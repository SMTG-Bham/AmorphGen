#!/usr/bin/env bash
# Shared by generated jobs and the historical BlueBEAR examples. Linux/Slurm
# supplies setsid and ps; no Python or cluster modules are needed to supervise.

amorphgen_slurm_main() {
    if [[ ${_AMORPHGEN_SLURM_WORKER:-0} == 1 ]]; then
        # A trapped (rather than ignored) signal is reset for exec'd programs,
        # allowing the Python checkpoint handler to install its own policy.
        _amorphgen_slurm_worker_stop() {
            local child_status=$1
            # Bash defers this trap until a foreground program exits. Keep a
            # real checkpoint/write failure distinct from cooperative exit75.
            # An interrupted asynchronous wait (138/143) is unconfirmed and
            # must never authorize an automatic requeue either.
            [[ $child_status != 0 ]] || child_status=75
            exit "$child_status"
        }
        trap '_amorphgen_slurm_worker_stop "$?"' USR1 TERM
        export AMORPHGEN_CHECKPOINT_ON_SIGNAL=1 PYTHONUNBUFFERED=1
        : > "${_AMORPHGEN_SLURM_READY:?}"
        # sbatch normally exports its caller's environment. A nested submission
        # must start its own supervisor instead of inheriting this handoff.
        unset _AMORPHGEN_SLURM_WORKER _AMORPHGEN_SLURM_READY
        return 0
    fi

    local script=$1
    shift
    local worker='' requested=0 sent=0 terminated=0 status=0
    local -a checkpoint_workers=()
    local ready
    ready=$(mktemp "${TMPDIR:-/tmp}/amorphgen-slurm.XXXXXXXX")
    rm -f "$ready"

    _amorphgen_slurm_forward() {
        [[ $requested == 1 && $sent == 0 && -n $worker && -f $ready ]] || return 0
        sent=1
        printf 'AmorphGen: checkpoint requested; waiting for workers to finish.\n' >&2
        local pid command
        # Stop the command sequence before enumerating children: a very fast
        # checkpoint must not let the shell launch the next scientific stage.
        kill -USR1 "$worker" 2>/dev/null || true
        # Do not signal pipeline helpers (tee, time) or kill the shell while it
        # is waiting: those must stay alive until checkpoint writes finish.
        while read -r pid command; do
            case "$command" in
                python*|amorphgen*)
                    checkpoint_workers+=("$pid")
                    kill -USR1 "$pid" 2>/dev/null || true
                    ;;
            esac
        done < <(ps -s "$worker" -o pid=,comm=)
    }
    _amorphgen_slurm_signal() {
        requested=1
        [[ $1 != TERM ]] || terminated=1
        _amorphgen_slurm_forward
    }
    trap '_amorphgen_slurm_signal USR1' USR1
    trap '_amorphgen_slurm_signal TERM' TERM
    # A new session groups nested Python commands, pipelines and time wrappers
    # without risking signals to the submission shell or another array task.
    _AMORPHGEN_SLURM_WORKER=1 _AMORPHGEN_SLURM_READY="$ready" \
        setsid bash "$script" "$@" &
    worker=$!
    while [[ ! -f $ready ]] && kill -0 "$worker" 2>/dev/null; do
        sleep 0.02
    done
    _amorphgen_slurm_forward
    while true; do
        if wait "$worker"; then status=0; else status=$?; fi
        # bash wait returns early when a trap runs. The child is still alive
        # while it writes its checkpoint; wait again instead of killing it.
        if [[ $requested == 1 && $status -gt 128 ]]; then
            if wait "$worker"; then status=0; else status=$?; fi
        fi
        kill -0 "$worker" 2>/dev/null || break
    done
    # If the job used an explicit background command plus bash's wait builtin,
    # the worker shell's trap can run before that command has finished. These
    # descendants are not our direct children, so poll their process state.
    local checkpoint_pid state
    for checkpoint_pid in "${checkpoint_workers[@]}"; do
        while kill -0 "$checkpoint_pid" 2>/dev/null; do
            state=$(ps -p "$checkpoint_pid" -o stat=) || break
            [[ $state != Z* ]] || break
            sleep 0.02
        done
    done
    rm -f "$ready"
    if [[ $requested == 1 ]]; then
        if [[ ${AMORPHGEN_SLURM_REQUEUE:-0} == 1 && $terminated == 0 && $status == 75 ]]; then
            local job=${SLURM_JOB_ID:?A Slurm job ID is required to requeue}
            if [[ -n ${SLURM_ARRAY_JOB_ID:-} && -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
                job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            fi
            scontrol requeue "$job" || { printf 'AmorphGen: requeue failed for %s.\n' "$job" >&2; exit 75; }
        fi
        # afterok dependants must never see a partially completed job succeed.
        # Preserve real worker errors; interrupted asynchronous waits still
        # use the checkpoint-request exit code, without authorizing requeue.
        case "$status" in
            0|75|138|143) exit 75 ;;
            *) exit "$status" ;;
        esac
    fi
    exit "$status"
}

amorphgen_slurm_bluebear_environment() {
    local bear_module=$1 python_module=$2
    if [[ ${AMORPHGEN_SKIP_MODULES:-0} != 1 ]]; then
        module purge
        module load "${AMORPHGEN_BEAR_MODULE:-$bear_module}"
        module load "${AMORPHGEN_PYTHON_MODULE:-$python_module}"
    fi
    source "${AMORPHGEN_VENV:?Set AMORPHGEN_VENV to your virtualenv directory}/bin/activate"
    export PYTHONUNBUFFERED=1
}
