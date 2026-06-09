import psutil

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

UCM_THREAD_PREFIX = "ucm_"


def _patched_init(original_init):
    def init(self, rank_id):
        original_init(self, rank_id)
        self.assign_ucm: dict[int, list[int]] = {}

    return init


def _patched_allocate(original_allocate):
    def allocate(self):
        original_allocate(self)
        for npu in list(self.assign_main.keys()):
            main_cpus = self.assign_main[npu]
            n_ucm = len(main_cpus) // 2
            if n_ucm == 0:
                self.assign_ucm[npu] = []
                continue
            self.assign_ucm[npu] = main_cpus[-n_ucm:]
            self.assign_main[npu] = main_cpus[:-n_ucm]
            logger.info(
                "NPU%s CPU split: main=%d CPUs, ucm=%d CPUs",
                npu,
                len(self.assign_main[npu]),
                n_ucm,
            )

    return allocate


def _patched_get_threads_map(original_get_threads_map):
    def get_threads_map(thread_message: str) -> dict[str, dict[str, list[str]]]:
        threads_map: dict[str, dict[str, list[str]]] = {}
        for line in thread_message.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            main_pid, sub_pid = parts[0], parts[1]
            thread_name = parts[-1] if len(parts) >= 4 else ""
            if "acl_thread" in line:
                key = "acl_thread"
            elif "release_thread" in line:
                key = "release_thread"
            elif thread_name.startswith(UCM_THREAD_PREFIX):
                key = "ucm_thread"
            else:
                continue
            if main_pid not in threads_map:
                threads_map[main_pid] = {
                    "acl_thread": [],
                    "release_thread": [],
                    "ucm_thread": [],
                }
            threads_map[main_pid][key].append(sub_pid)
        return threads_map

    return get_threads_map


def _patched_bind_threads(original_bind_threads, execute_command_fn):
    def bind_threads(self):
        thread_message, _ = execute_command_fn(["ps", "-Te"])
        threads_map = self.get_threads_map(thread_message)
        main_pid = str(psutil.Process().pid)
        current_npu = self.device_info.running_npu_list[self.rank_id]

        self.bind(main_pid, self.assign_main[current_npu], True)
        for acl_thread in threads_map.get(main_pid, {}).get("acl_thread", []):
            self.bind(acl_thread, self.assign_acl[current_npu], False)
        for release_thread in threads_map.get(main_pid, {}).get("release_thread", []):
            self.bind(release_thread, self.assign_rel[current_npu], False)

        ucm_threads = threads_map.get(main_pid, {}).get("ucm_thread", [])
        ucm_cpus = self.assign_ucm.get(current_npu, [])
        if ucm_threads and ucm_cpus:
            logger.info(
                "[ucm_bind] Binding %d UCM threads to CPUs [%s].",
                len(ucm_threads),
                " ".join(map(str, ucm_cpus)),
            )
            for ucm_tid in ucm_threads:
                self.bind(ucm_tid, ucm_cpus, False)
        elif ucm_threads:
            logger.warning(
                "UCM threads detected (%d) but no dedicated CPU pool. "
                "UCM threads will share inference CPUs on assign_main.",
                len(ucm_threads),
            )

        self.bind_memory(main_pid, current_npu)

    return bind_threads


def _patched_print_plan(original_print_plan):
    def print_plan(self):
        from vllm.logger import logger as vllm_logger

        vllm_logger.info("The CPU allocation plan is as follows:")
        current_npu = self.device_info.running_npu_list[self.rank_id]
        main = " ".join(map(str, self.assign_main[current_npu]))
        acl = " ".join(map(str, self.assign_acl[current_npu]))
        rel = (
            str(self.assign_rel[current_npu])
            if self.assign_rel[current_npu]
            else ""
        )
        ucm = (
            " ".join(map(str, self.assign_ucm.get(current_npu, [])))
            if self.assign_ucm.get(current_npu, [])
            else ""
        )
        vllm_logger.info(
            "NPU%s: main=[%s]  acl=[%s]  release=[%s]  ucm=[%s]",
            current_npu,
            main,
            acl,
            rel,
            ucm,
        )

    return print_plan


@when_imported("vllm_ascend.cpu_binding")
def patch_cpu_binding(mod):
    cls = mod.CpuAlloc

    cls.__init__ = _patched_init(cls.__init__)
    cls.allocate = _patched_allocate(cls.allocate)
    cls.get_threads_map = staticmethod(
        _patched_get_threads_map(cls.get_threads_map)
    )
    cls.bind_threads = _patched_bind_threads(
        cls.bind_threads, mod.execute_command
    )
    cls.print_plan = _patched_print_plan(cls.print_plan)

    logger.info("UCM cpu_binding patch applied: main/ucm 50/50 split")