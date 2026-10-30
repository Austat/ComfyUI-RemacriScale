import os, queue, threading, time
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar
from tqdm import tqdm
import folder_paths
from .cache import RuntimeCacheMixin
from .inference import InferenceMixin
from .model import ModelMixin
from .session import SessionMixin

@dataclass
class DeviceStats:
    device_id: int
    name: str
    provider: str
    images: int = 0
    batches: int = 0
    prepare_seconds: float = 0.0
    inference_seconds: float = 0.0
    post_seconds: float = 0.0
    worker_seconds: float = 0.0
    errors: int = 0

class RemacriOnnxUpscaleNode(RuntimeCacheMixin, ModelMixin, SessionMixin, InferenceMixin):
    RUNTIME_VERSION_FILE = "./trt_cache_metadata/runtime_versions.json"
    TRT_ENGINE_CACHE_PATH = "./trt_engine_cache"
    TRT_TIMING_CACHE_PATH = "./trt_timing_cache"
    _sessions, _active_providers = {}, {}
    _session_lock = threading.Lock()
    _metadata_cache: Dict[str, Tuple[str, Tuple, str, Tuple]] = {}

    @staticmethod
    def _device_choices():
        choices = ["Auto"]
        if torch.cuda.is_available():
            choices += [f"Device {i}: {torch.cuda.get_device_name(i)}"
                        for i in range(torch.cuda.device_count())]
            if torch.cuda.device_count() > 1:
                ids = ':'.join(map(str, range(torch.cuda.device_count())))
                choices.append(f"Device {ids}: All GPUs")
        else:
            choices.append("CPU")
        return choices

    @classmethod
    def INPUT_TYPES(cls):
        files=[]
        for directory in folder_paths.get_folder_paths("upscale_models"):
            if os.path.isdir(directory):
                files += [f for f in os.listdir(directory)
                          if f.lower().endswith('.onnx') and f not in files]
        return {"required": {
            "image": ("IMAGE",), "model_file": (sorted(files) or ["(no .onnx models found)"],),
            "provider": (["MIGraphXExecutionProvider", "ROCmExecutionProvider",
                          "TensorrtExecutionProvider", "CUDAExecutionProvider",
                          "CPUExecutionProvider"],),
            "devices": (cls._device_choices(),),
            "final_resolution": (["hd","fhd","2k","4k","8k","512x512","768x768",
                                  "1024x1024","2048x2048","4096x4096","no downscaling"],),
            "batch_size": ("INT", {"default":2,"min":1,"max":64,"step":1}),
            "keep_model_loaded": ("BOOLEAN", {"default":True}),
            "diagnostics": ("BOOLEAN", {"default":False})},
            "optional": {"device_ids": ("STRING", {"default":""})}}

    RETURN_TYPES=("IMAGE", "STRING"); RETURN_NAMES=("upsampled", "upscale_info"); FUNCTION="upscale"
    CATEGORY="image/upscale"; OUTPUT_NODE=True

    @staticmethod
    def _parse_devices(devices, device_ids, provider):
        if provider == "CPUExecutionProvider": return [None]
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("GPU provider selected but PyTorch exposes no CUDA/ROCm GPUs.")
        text=(device_ids or '').strip() or str(devices).strip()
        if text.lower()=="auto": return list(range(torch.cuda.device_count()))
        if text.lower().startswith("device "):
            text=text[7:].split(' ',1)[0].rstrip(':')
        result=[]
        for token in text.replace(', ',':').replace(',',':').replace(';',':').split(':'):
            if not token: continue
            did=int(token)
            if not 0 <= did < torch.cuda.device_count():
                raise ValueError(f"GPU {did} does not exist; valid range 0..{torch.cuda.device_count()-1}")
            if did not in result: result.append(did)
        if not result: raise ValueError("No GPU devices selected")
        return result

    @staticmethod
    def _summary(stats,total,wall,diagnostics):
        done=sum(s.images for s in stats.values())
        print("\n[RemacriOnnxUpscale] Multi-device processing summary")
        for did in sorted(stats, key=lambda x:-1 if x is None else x):
            s=stats[did]; share=100*s.images/total if total else 0
            speed=s.images/s.inference_seconds if s.inference_seconds else 0
            label="CPU" if did is None else f"GPU {did} {s.name}"
            print(f"[RemacriOnnxUpscale] {label}: {s.images}/{total} ({share:.2f} %), "
                  f"worker={s.worker_seconds:.3f}s, inference={s.inference_seconds:.3f}s, "
                  f"throughput={speed:.2f} images/s")
            if diagnostics:
                print(f"[RemacriOnnxUpscale]   provider={s.provider}, batches={s.batches}, "
                      f"prepare={s.prepare_seconds:.3f}s, post={s.post_seconds:.3f}s, errors={s.errors}")
        print(f"[RemacriOnnxUpscale] Total: {done}/{total}, wall={wall:.3f}s, "
              f"accounting={'OK' if done==total else 'ERROR'}")

    def upscale(self,image,model_file,provider,devices,final_resolution,batch_size,
                keep_model_loaded=True,diagnostics=False,device_ids="",progress=None):
        wall=time.perf_counter(); model=self._find_model(model_file)
        if image.dim()==3: image=image.unsqueeze(0)
        if image.dim()!=4: raise ValueError(f"Expected NHWC IMAGE, got {tuple(image.shape)}")
        total,h,w,c=image.shape; batch_size=max(1,min(int(batch_size),total))
        self._check_runtime_versions_and_invalidate_cache()
        input_name,_,output_name,_=self._validate_model_input(model,batch_size,c,h,w)
        selected=self._parse_devices(devices,device_ids,provider)
        timing=os.path.join(self.TRT_TIMING_CACHE_PATH,f"timing_{h}x{w}_b{batch_size}.bin")
        sessions={}; stats={}
        for did in selected:
            sess,active=self._load_session(model,provider,input_name,batch_size,c,h,w,
                                           timing,diagnostics,did)
            sessions[did]=(sess,active)
            name="CPU" if did is None else torch.cuda.get_device_name(did)
            stats[did]=DeviceStats(-1 if did is None else did,name,active)
        print("[RemacriOnnxUpscale] Selected: "+", ".join(
              "CPU" if d is None else f"GPU {d} {torch.cuda.get_device_name(d)}" for d in selected))
        jobs=queue.Queue(); target=self._target_size(final_resolution)
        for start in range(0,total,batch_size): jobs.put((start,min(start+batch_size,total)))
        outputs=[None]*((total+batch_size-1)//batch_size); lock=threading.Lock(); errors=[]
        comfy_pbar=ProgressBar(total)
        terminal_pbar=tqdm(total=total,desc="Remacri multi-GPU upscale",unit="image",dynamic_ncols=True)
        processed=0
        upscaled_hw=None
        def worker(did):
            nonlocal processed, upscaled_hw
            sess,active=sessions[did]; s=stats[did]
            direct=did is not None and active in ("CUDAExecutionProvider","TensorrtExecutionProvider")
            device=torch.device("cuda",did) if direct else torch.device("cpu")
            begin=time.perf_counter()
            try:
                if did is not None: torch.cuda.set_device(did)
                while True:
                    try: start,end=jobs.get_nowait()
                    except queue.Empty: break
                    try:
                        t=time.perf_counter()
                        inp=image[start:end].detach().permute(0,3,1,2).contiguous().to(device,dtype=torch.float32)
                        if direct: torch.cuda.synchronize(did)
                        s.prepare_seconds+=time.perf_counter()-t; t=time.perf_counter()
                        out=(self._run_gpu_iobinding(sess,input_name,output_name,inp)
                             if direct else self._run_numpy(sess,input_name,inp))
                        current_upscaled_hw=(int(out.shape[-2]), int(out.shape[-1]))
                        with lock:
                            if upscaled_hw is None:
                                upscaled_hw=current_upscaled_hw
                            elif upscaled_hw != current_upscaled_hw:
                                raise RuntimeError(
                                    f"Inconsistent model output sizes: {upscaled_hw} and {current_upscaled_hw}"
                                )
                        if direct: torch.cuda.synchronize(did)
                        s.inference_seconds+=time.perf_counter()-t; t=time.perf_counter()
                        out=torch.nan_to_num(out,nan=0,posinf=1,neginf=0).clamp_(0,1)
                        if target and tuple(out.shape[-2:])!=target:
                            out=F.interpolate(out,size=target,mode="area")
                        out=out.permute(0,2,3,1).contiguous().cpu()
                        s.post_seconds+=time.perf_counter()-t
                        with lock:
                            outputs[start//batch_size]=out; n=end-start
                            s.images+=n; s.batches+=1; processed+=n; comfy_pbar.update(n); terminal_pbar.update(n)
                    finally: jobs.task_done()
            except Exception as exc:
                s.errors+=1
                with lock: errors.append((did,exc))
            finally: s.worker_seconds=time.perf_counter()-begin
        threads=[threading.Thread(target=worker,args=(d,)) for d in selected]
        for t in threads:t.start()
        for t in threads:t.join()
        if errors: raise RuntimeError(f"Worker GPU {errors[0][0]} failed: {errors[0][1]}")
        if any(x is None for x in outputs): raise RuntimeError("One or more output batches are missing")
        result=torch.cat(outputs,dim=0)
        self._summary(stats,total,time.perf_counter()-wall,diagnostics)
        final_h,final_w=int(result.shape[1]),int(result.shape[2])
        upscale_h,upscale_w=upscaled_hw or (final_h,final_w)
        info=(f"Original input resolution: {h} x {w}\n"
              f"Model upscale resolution: {upscale_h} x {upscale_w}\n"
              f"Final downscaled resolution: {final_h} x {final_w}")
        print(f"[RemacriOnnxUpscale] Resolution summary:\n{info}")
        if not keep_model_loaded:self.__class__._clear_sessions()
        return (result,info)
