import multiprocessing as mp
import setproctitle
import faulthandler
import psutil
import signal



from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from typing import Optional
from sglang.srt.utils import (
    kill_itself_when_parent_died,
    set_gpu_proc_affinity,
    get_bool_env_var,
    configure_logger,
    suppress_other_loggers,
)
from sglang.srt.managers.scheduler import Scheduler, DisaggregationMode
from sglang.utils import get_exception_traceback



def run_migrate_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Generate the prefix
    prefix = "migrate"
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"

    # Config the process
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"sglang::scheduler_{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # Configure the logger for this process only
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank)
        # Sent to the main process that the scheduler is ready. 
        print(f"[DEBUG scheduler.py] Migration Scheduler is piping initialisation information to main process.")
        pipe_writer.send(
            {
                "status": "migrate_ready",
            }
        )
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode

        if disaggregation_mode == DisaggregationMode.NULL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        

def launch_migration_scheduler_process(server_args: ServerArgs):
        "Starts a migration scheduler process to handle live migration."
        reader_mig, writer_mig = mp.Pipe(duplex=False)
        gpu_id_mig = 1 # Hardcoded to migrate to GPU 1 for now
        # Hardcode tp_rank and pp_rank to 0 for migration scheduler for now
        tp_rank = 0
        pp_rank = 0
        print(f"[DEBUG tokenizer_manager.py] ------------------------------------ Launching migration scheduler process on GPU {gpu_id_mig} ------------------------------------")
        port_args_mig = PortArgs.init_new(server_args)
        print(f"[DEBUG tokenizer_manager.py] Created migration scheduler with ipc filename {port_args_mig.scheduler_input_ipc_name}")
        
        proc_mig = mp.Process(
            target = run_migrate_scheduler_process,
            args=(
                server_args,
                port_args_mig,
                gpu_id_mig,
                tp_rank,
                pp_rank,
                None,
                writer_mig,
            ),
        )
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        
        with memory_saver_adapter.configure_subprocess():
            proc_mig.start()
        
        return proc_mig, reader_mig