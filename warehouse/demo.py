import argparse
from contextlib import nullcontext
from queue import Empty, Queue
import sys
from threading import Thread
import time

from warehouse.contracts import OperatorEvent, State
from warehouse.evaluate import Evaluator
from warehouse.runtime import Experiment, SCENARIOS, validate_modes


HELP = "solicitar SKU | confirmar | estado | pausa | continuar | reset | salir | ayuda"
BANNER = (
    "ALMACEN DEMO | Pose: ORACLE | Transporte: IDEALIZED (acoplamiento cinematico, SIN validacion fisica)\n"
    "Stock: VISION_AREA, nivel visible de parches; no unidades ni cambios de masa.\n"
    "CONTACT y localizacion VISION pendientes. Una solicitud por ciclo; reset completo explicito, sin reanudacion."
)


def terminal_reader(commands):
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put("stdin_closed")


def show_status(experiment):
    controller = experiment.controller
    print(f"Estado: {controller.state} | t_sim={experiment.sim.data.time:.2f}s | motivo={controller.failure_reason or '-'}")
    for bin in experiment.inventory.bins.values():
        observation = experiment.inventory.observations.get(bin.id)
        stock = f"{observation.level} ({observation.source}, t_sim={observation.timestamp:.2f}s; ultima observacion, no tiempo real)" if observation else "UNKNOWN (sin observacion)"
        print(f"  {bin.sku:10} {bin.id} origen={bin.home_slot_id} ubicacion={bin.current_location} {stock}")
    for alert in experiment.inventory.alerts.values():
        if alert["active"]:
            print(f"  AVISO LOCAL: reponer {alert['sku']} / {alert['bin_id']}: {alert['level']} desde t_sim={alert['first_seen_sim_s']:.2f}s")
    if controller.state == State.WAIT_OPERATOR:
        print("Accion humana pendiente: 'confirmar' representa retirar material y cambia el contenido visual.")


def viewer_for(experiment, commands, headless):
    if headless:
        return nullcontext(None)
    import mujoco.viewer

    def keypress(key):
        command = {67: "confirmar", 80: "pausa", 82: "continuar", 81: "salir"}.get(key)
        if command:
            commands.put(command)

    viewer = mujoco.viewer.launch_passive(experiment.sim.model, experiment.sim.data, key_callback=keypress)
    with viewer.lock():
        viewer.cam.lookat[:] = (0.55, -0.05, 0.45)
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -25
    return viewer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Warehouse mini demo; explicitly idealized transport")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perception", choices=("oracle", "vision"), default="oracle")
    parser.add_argument("--transport", choices=("idealized", "contact"), default="idealized")
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--sku")
    parser.add_argument("--auto", action="store_true", help="Use an explicitly simulated operator and exit after one episode")
    parser.add_argument("--headless", action="store_true", help="Disable native viewer; camera rendering remains enabled")
    parser.add_argument("--output", default="runs")
    args = parser.parse_args(argv)
    try:
        validate_modes(args.perception, args.transport)
    except ValueError as error:
        parser.error(str(error))
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    commands = Queue()
    if not args.auto:
        Thread(target=terminal_reader, args=(commands,), daemon=True).start()
    print(BANNER, flush=True)
    print(f"Operador: {'SIMULATED' if args.auto else 'HUMAN'} | seed={args.seed} | escenario={args.scenario}")
    print(HELP)
    reset = True
    exit_code = 0
    while reset:
        reset = False
        with Experiment(args.seed, args.perception, args.transport, args.scenario, args.output,
                        "SIMULATED" if args.auto else "HUMAN") as experiment:
            controller = experiment.controller
            evaluator = Evaluator()
            saved = False
            paused = False
            last_state = None
            last_render = 0.0
            show_status(experiment)
            if args.sku or args.auto:
                try:
                    experiment.start(args.sku or "TORNILLOS")
                except ValueError as error:
                    parser.error(str(error))
            print(f"Evidencia: {experiment.log.path}", flush=True)
            try:
                with viewer_for(experiment, commands, args.headless) as viewer:
                    running = True
                    while running:
                        started = time.monotonic()
                        events = []
                        while True:
                            try:
                                command = commands.get_nowait()
                            except Empty:
                                break
                            parts = command.split()
                            if not parts:
                                continue
                            action = parts[0].lower()
                            try:
                                if action == "solicitar" and len(parts) == 2:
                                    experiment.start(parts[1].upper())
                                elif action == "confirmar":
                                    if controller.request and not paused and controller.state == State.WAIT_OPERATOR:
                                        events.append(OperatorEvent(controller.request.id))
                                    else:
                                        print("Confirmacion ignorada: requiere WAIT_OPERATOR sin pausa.")
                                elif action in ("pausa", "continuar"):
                                    paused = action == "pausa"
                                    controller.event("SIMULATION_PAUSED" if paused else "SIMULATION_RESUMED", experiment.sim.snapshot())
                                    print("Simulacion pausada (no es un paro industrial)." if paused else "Simulacion reanudada.")
                                elif action in ("reset", "salir"):
                                    controller.event("RESET_REQUESTED" if action == "reset" else "EXIT_REQUESTED", experiment.sim.snapshot())
                                    if controller.request and not controller.done:
                                        controller.fail("EXPLICIT_SCENE_RESET" if action == "reset" else "OPERATOR_ABORT", experiment.sim.snapshot())
                                    reset = action == "reset"
                                    running = False
                                    break
                                elif action == "estado":
                                    show_status(experiment)
                                elif action == "stdin_closed":
                                    print("Entrada cerrada; utiliza teclas C confirmar, P pausa, R continuar, Q salir en visor.")
                                    if args.headless:
                                        if controller.request:
                                            controller.fail("OPERATOR_INPUT_CLOSED", experiment.sim.snapshot())
                                        running = False
                                        exit_code = 1
                                else:
                                    print(HELP)
                            except ValueError as error:
                                print(f"Solicitud rechazada: {error}")
                        if running and not paused:
                            experiment.tick(experiment.simulated_operator_events() if args.auto else events)
                            evaluator.sample(experiment)
                        if controller.state != last_state:
                            last_state = controller.state
                            show_status(experiment)
                        if controller.done and not saved:
                            result = evaluator.result(experiment)
                            experiment.log.write_json("result.json", result)
                            saved = True
                            exit_code = 0 if result["success"] else 1
                            print(f"Resultado: success={result['success']} | motivo={result['failure_reason']} | logs={experiment.log.path}", flush=True)
                        if args.auto and controller.done:
                            running = False
                        if viewer and time.monotonic() - last_render >= 1 / 25:
                            viewer.sync()
                            last_render = time.monotonic()
                            if not viewer.is_running():
                                commands.put("salir")
                        if not args.headless or not args.auto:
                            time.sleep(max(0, 0.02 - (time.monotonic() - started)))
            except KeyboardInterrupt:
                if controller.request and not controller.done:
                    controller.fail("OPERATOR_INTERRUPT", experiment.sim.snapshot())
                    experiment.log.write_json("result.json", evaluator.result(experiment))
                return 130
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
