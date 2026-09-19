# THEKER — kits para montaje de reductores

> **Implementación principal: preparación y entrega de kits de componentes.**
> La sección [Demo THEKER](#demo-theker--preparación-de-pedido) describe la
> nueva implementación (`orderpick/`); el resto del documento es el
> prototipo anterior (`warehouse/`), conservado como referencia ejecutable.

**Tarea humana elegida:** un operario consulta una orden, recorre el supermercado
de componentes, cuenta referencias, prepara una bandeja, retira el recipiente
agotado y entrega el kit al puesto de montaje de reductores industriales.
`orderpick/` automatiza ese ciclo en simulación. **No ensambla el reductor.**
El kit de demostración no es una lista de materiales de un producto comercial;
la aplicación y el utillaje deben contrastarse con THEKER.

---

## Demo THEKER — preparación de pedido

Brazo **Franka Emika Panda** (MuJoCo Menagerie, commit `8161bba`, assets
vendorizados con licencia) con pinza paralela, montado sobre un **carro
actuado en un carril** frente a una estantería de 3 slots. El carro lleva
además la bandeja de pedido (3 compartimentos, labios de contención, asa
tipo "sillín") y una **cámara de muñeca** calibrada (RGB-D).

Ciclo sin intervención humana:

1. Pedido `2×ENGRANAJE + 1×RODAMIENTO + 1×ESPARRAGO`.
2. El carro se desplaza a cada slot (actuador, no teletransporte).
3. Percepción: la cámara de muñeca renderizada segmenta los marcadores de
   color de las piezas y retro-proyecta su posición con profundidad
   métrica y extrínsecos de la cámara (`--perception vision`). El modo
   `--perception oracle` lee la pose verdadera y está **etiquetado como
   depuración**. No hay fallback de localización de producto a ORACLE.
   **Ambos modos siguen usando verdad para recipientes, bandeja y guard de
   deslizamiento**: la percepción del modo visual es híbrida.
4. Pinza física por contacto sobre el cuello de cada pieza; verificación
   por encoder de pinza y por re-observación (la unidad objetivo debe
   desaparecer del recipiente).
5. Depósito en la bandeja a bordo, con compartimentos balanceados.
6. Si el recipiente frontal queda vacío: se retira al aparcamiento de
   vacíos (verificando antes que está libre) y se **adelanta la reserva**
   físicamente enganchando el recipiente posterior.
7. El carro lleva la bandeja a recepción (bandeja estable en el bolsillo);
   se verifica que la mesa está libre, se recoge la bandeja por el asa
   (soporte mecánico por compresión sobre las almohadillas) y se deposita.
8. Evaluación independiente: contenido exacto por SKU, bandeja dentro de la
   mesa, orientación, contacto de soporte, ausencia de contacto con pinza,
   piezas apoyadas en la bandeja y estabilidad durante 0,5 s. Un pedido
   parcial, con sobrantes o sostenido por la pinza **no es un éxito**.

Una observación fuera de cámara o con profundidad ilegible queda `unknown`.
El frontal solo se retira tras confirmar su vacío; agotar reintentos de agarre
no confirma agotamiento de stock.

```bash
MUJOCO_GL=egl .venv/bin/python -m orderpick.demo --seed 7 --perception vision --headless
MUJOCO_GL=egl .venv/bin/python -m orderpick.evaluate --seeds 7:8 --modes oracle,vision --scenarios nominal

# Otra receta con el mismo catálogo, sin cambiar el controlador
MUJOCO_GL=egl .venv/bin/python -m orderpick.demo --headless --perception vision \
  --recipe config/recipe-service.json
```

En `orderpick.evaluate`, `7:8` incluye ambas semillas; los rangos del prototipo
legacy tienen otra semántica. Un lote con cualquier entrega no exacta devuelve
código 1, incluso si contiene escenarios de fallo intencionado.

`--recipe` acepta JSON con `id`, `workstation` y `lines: [{"sku": "...", "qty": 1}]`.
Se rechazan referencias inexistentes, cantidades no enteras/positivas y líneas
duplicadas. Cambiar la receta no aumenta el stock ni modifica la escena.

`--video` guarda una **secuencia PNG**, no vídeo codificado, de `overview`.
`--run-dir` es la carpeta padre: siempre se crea una subcarpeta única.
`metadata.json` identifica la receta, ayudas oraculares y hash de fuentes
incluyendo `orderpick/`; `result.json` guarda comprobación física y eventos.
Sin `--headless` se abre el visor nativo; no usar `MUJOCO_GL=egl` para ese modo.

### Auditoría y medición

Base auditada: `c60c429`, rama `devin/theker-order-picking`.
Sus 39 tests pasaron, pero el benchmark nominal seed 7 entregó **0/1 pedidos
exactos en oracle y 0/1 en visión** con su evaluador original. La afirmación
anterior de una cadena física completa no se reprodujo en esa revisión.

Localización RGB-D en los 3 bins frontales, semillas 7, 8 y 9 (9 observaciones,
conteos correctos en ambas versiones):

| Error XY | Base `c60c429` | Antes de integrar la base (`ea10ac2`) |
|---|---:|---:|
| Media | 16,739 mm | 2,946 mm |
| Máximo | 20,787 mm | 7,044 mm |

La media baja un 82,4 %. **Todavía no cumple el objetivo de máximo 5 mm**:
la herramienta de calibración termina con código 1. Esto mide localización
de piezas visibles, no fiabilidad del ciclo completo ni escenas ocluidas.

```bash
MUJOCO_GL=egl .venv/bin/python -m orderpick.calibrate --seeds 7 8 9
```

Para repetir la comparación, ejecutar el mismo script de calibración desde
un checkout de `c60c429`, con `PYTHONPATH=.` y ruta absoluta al script de esta
rama. Así se utilizan las clases del checkout base y la misma medición.

Se reprodujeron cuatro fallos relevantes: región fuera de cámara declarada
vacía, error visual de aproximadamente 20 mm en Y, descenso de la pinza
contra las paredes de bandeja, y entrega declarada aunque la bandeja siguiera
apoyada en la pinza. La localización ahora usa puntos RGB-D de la superficie
superior; la descarga respeta la altura de pared y retira la pinza del asa.
La validación física permanece más exigente que la señal `delivered` del
controlador. Consultar `assessment` y `exact`, no solo esa señal.

El benchmark conserva **todos** los episodios, incluidos fallos. Los cambios
en verificación impiden comparar porcentajes antiguos y nuevos sin indicar
qué criterio se utilizó. Una muestra pequeña tampoco demuestra fiabilidad
industrial.

La base `74e05b2` amplía la bandeja, separa los huecos de descarga y mejora
el agarre y la espera de asentamiento. Su documentación declara entregas
completas en visión (seed 7) y oracle (7 y 12) con su criterio anterior.
Estas mejoras se conservan, pero las entregas deben medirse de nuevo con
el evaluador físico independiente antes de atribuirlas a esta integración.

### Hipótesis y límites

- Las piezas llevan marcadores de color declarados en la cabeza; la
  percepción es por color+profundidad calibrada, **no** reconocimiento
  general de productos.
- Las bandas HSV están en `config/catalog.json`. Añadir una referencia
  necesita calibración de su marcador, geometría y bin, no solo un nombre.
- Las geometrías son peones con cuello y cabeza adaptados a la pinza, no
  engranajes/rodamientos/espárragos industriales completos. Algunas superficies
  son visuales; los contactos usan geometrías simplificadas. La masa efectiva
  incluye geometrías auxiliares.
- La variabilidad efectiva es jitter X hasta 18 mm, Y hasta 4 mm (puede ser
  cero por las holguras del bin), yaw hasta 0,5 rad y masa nominal ±15 %.
  No se ha implementado variación de tamaño ni deformación. Los umbrales de
  visibilidad son heurísticos; no certifican ausencia de oclusiones pequeñas.
- Recipientes, bandeja y monitor de deslizamiento usan poses oraculares.
  La bandeja puede moverse y caer: no equivale a una pose fija calibrada.
- `obs_glitch`/`obs_occluded` inyectan fotogramas corruptos/ausentes en la
  cámara de muñeca; `grasp_slip` pulimenta la fricción del post de una
  pieza; `park_blocked`/`reception_blocked` colocan un obstáculo magenta
  físico sobre la superficie.
- Sin ROS, servicios, LLM ni frontend; un proceso posee la simulación.
- No hay validación con hardware, reconocimiento general de piezas, planificación
  general de colisiones ni recuperación tras reiniciar el proceso.

### Prioridades siguientes para competir

1. Medir repetibilidad del kit completo y el fallo físico dominante antes de
   ampliar la escena. Publicar entregas exactas, parciales, caídas y tiempos.
2. Sustituir las poses oraculares de logística por percepción de recipientes
   y bandeja y sensores de agarre. Mantener la verdad solo en evaluación.
3. Validar con THEKER la tarea humana, tiempos, BOM y utillaje; sustituir peones
   por geometrías reales y medir holguras y variabilidad efectiva.
4. Demostrar otra receta, una perturbación y una parada por observación
   desconocida. Los fallos gestionados no se cuentan como pedidos entregados.
5. Llevar al jurado una diapositiva con tarea, ciclo, métricas y límites; un
   vídeo de respaldo no sustituye la ejecución en directo.

### Verificación de desarrollo

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest -q
uv tool run --from ruff==0.12.12 ruff check --select E9,F63,F7,F82 \
  orderpick tests/test_orderpick_regressions.py warehouse/persistence.py
uv tool run --from mypy==1.17.1 mypy --ignore-missing-imports --follow-imports=silent \
  orderpick/orders.py orderpick/verification.py
```

El chequeo de tipos cubre los nuevos módulos de recetas y verificación; no
equivale a tipar todo el controlador ni las librerías nativas.

---

## Prototipo legacy: qué está entregado

Ciclo completo en **ORACLE + IDEALIZED**: reservar recipiente y hueco → aproximar → observar pose → alinear → acoplar y extraer → entregar y desacoplar en recepción → esperar confirmación → cambiar contenido visual → estimar nivel desde una imagen → generar alerta local → **volver a recoger en recepción** → devolver → verificar → liberar reserva.

- MuJoCo, ejes cartesianos X/Y/Z actuados, estantería de 2 columnas × 2 alturas y recepción alineada.
- Tres SKUs (`TORNILLOS`, `TUERCAS`, `ARANDELAS`), tres recipientes iguales; solo uno activo por ciclo. Las tres ubicaciones tienen pruebas de ciclo.
- Terminal de operador, visor nativo opcional, reset explícito y logs locales.
- Nivel de stock `EMPTY/LOW/OK/UNKNOWN` por segmentación de imágenes realmente renderizadas de la cámara superior. No hay contador validado: `count_estimate` siempre es `null`.
- Evaluación independiente de la pose, entrega, nueva recogida, retorno e integridad del inventario.

**No hay transporte CONTACT ni localización VISION.** Pedir cualquiera de esos modos termina con error, sin fallback silencioso. La cámara de stock no convierte la localización ORACLE en localización visual.

## Arranque

Ejecutar desde la raíz del repositorio. Entorno verificado: Linux, Python **3.12.14**, MuJoCo **3.3.7**, NumPy **2.2.6**, OpenCV **4.11.0**, pytest **8.4.2**. Las versiones de distribución, incluidas las transitivas, están en `requirements-lock.txt`.

Este equipo tiene Python 3.14 como `python3`; se ha usado Python 3.12 disponible mediante `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
source .venv/bin/activate
python -m warehouse.demo --seed 7 --perception oracle --transport idealized
```

Si `.venv` ya existe con las dependencias instaladas, basta con activarlo y ejecutar la última línea. No recrear el entorno durante la presentación.

Alternativa estándar para una máquina que ya tenga `python3.12` (no ejecutada en este equipo):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

No se necesitan servicios externos, claves, APIs ni conexión una vez instaladas las dependencias. En macOS, el visor pasivo requiere el lanzador `mjpython`; esa plataforma no se ha probado aquí.

### Operación interactiva

La demo arranca en `IDLE`. Escribir en la terminal:

```text
solicitar TORNILLOS
```

Al llegar a `WAIT_OPERATOR`, escribir:

```text
confirmar
```

Esto representa la retirada humana: reduce los parches visuales naranjas, sin comunicar su cantidad al estimador. En el escenario nominal, la imagen pasa de `OK` a `LOW`, aparece una alerta y la caja vuelve al hueco original. No hay retirada automática salvo que se solicite `--auto`.

| Comando | Acción |
|---|---|
| `solicitar SKU` | Iniciar la única solicitud de este ciclo |
| `confirmar` | Aceptar una sola retirada en `WAIT_OPERATOR`; duplicados se ignoran |
| `estado` | Catálogo, ubicación, estado, observación histórica y avisos |
| `pausa` / `continuar` | Pausar/reanudar la simulación, no un paro industrial |
| `reset` | Crear escena e inventario iniciales con la misma seed, conservando los logs anteriores |
| `salir` | Cerrar; una solicitud activa queda como incidente en los registros |
| `ayuda` | Mostrar comandos |

En el visor: **C** confirma, **P** pausa, **R** continúa y **Q** cierra. La terminal muestra modos y diagnóstico; el visor por sí solo no representa todo el estado operativo.

`--sku TORNILLOS` inicia la petición al abrir. Tras completar o fallar, usar `reset` para otro ciclo. Con `--sku`, el reset inicia otra petición de ese SKU. El reset **no es recuperación física**: sustituye toda la escena y crea otra carpeta de ejecución. No borra ni libera silenciosamente la reserva del registro anterior.

### Ejecuciones automáticas comprobadas

Con visor y operador explícitamente simulado:

```bash
python -m warehouse.demo --seed 7 --perception oracle --transport idealized --auto
```

Sin visor, conservando el renderizado de cámara:

```bash
MUJOCO_GL=egl python -m warehouse.demo --seed 7 --perception oracle --transport idealized --headless --auto
```

`--headless` también admite comandos por terminal si se omite `--auto`. La ejecución automática termina al cerrar el episodio; la interactiva espera `reset` o `salir`.

Se ha comprobado la apertura del visor y el renderizado de las cámaras frontal y superior por separado. En este equipo aparecieron avisos de EGL y decoraciones Wayland, pero las pruebas y el ciclo con visor terminaron correctamente. EGL depende del entorno gráfico; en una sesión con pantalla también se ha probado el renderizador predeterminado sin `MUJOCO_GL=egl`. Si falla la cámara con una excepción recuperable, el stock queda `UNKNOWN` con origen `CAMERA_UNAVAILABLE`; no se inventa una observación ni una alerta. Un fallo nativo del driver puede terminar el proceso.

## Evaluación y pruebas

```bash
MUJOCO_GL=egl python -m pytest -q

MUJOCO_GL=egl python -m warehouse.evaluate \
  --seeds 0:20 --perception oracle --transport idealized

MUJOCO_GL=egl python -m warehouse.evaluate \
  --seeds 0:3 --perception oracle --transport idealized \
  --scenarios lateral missing empty unknown_stock blocked_return pick_timeout
```

`START:STOP` excluye `STOP`. En `lateral`, `--offset-mm 10` aplica magnitud de 10 mm con signo elegido por la seed. `--offset-mm 0` conserva desplazamiento cero. También se han probado offsets exactos de ±5 y ±20 mm en pytest. **La corrección usa ORACLE**, no visión; no hay comparación baseline/visión entregada.

### Resultados ejecutados durante la implementación

Modos en todas las filas: **ORACLE + IDEALIZED**, operador **SIMULATED**. No son resultados de contacto físico.

| Escenario | Episodios | Ciclos completos | Excepciones esperadas bien manejadas |
|---|---:|---:|---:|
| Nominal, seeds 0–19 | 20 | 20 | 0 |
| Lateral ±10 mm, seeds 0–2 | 3 | 3 | 0 |
| Caja vacía | 3 | 3 | 0 |
| Imagen de stock desconocida | 3 | 3 | 0 |
| Caja ausente | 3 | 0 | 3 |
| Retorno bloqueado | 3 | 0 | 3 |
| Timeout de recogida | 3 | 0 | 3 |

En esos 38 episodios: **0 falsos éxitos y 0 falsos EMPTY**. Los escenarios nominales son deterministas y las distintas seeds no introducen perturbaciones: esos 20 ciclos prueban repetibilidad, no diversidad ni una tasa de fiabilidad industrial. Son pruebas de desarrollo, no un conjunto reservado tras congelar.

Pruebas automatizadas: **32 aprobadas** en la última ejecución registrada. Incluyen tres ciclos consecutivos desde reset, las otras dos ubicaciones del catálogo, offsets, doble confirmación, reservas en recepción y ante fallo, observación obsoleta, bloqueo de transporte con horquilla extendida, UNKNOWN, deduplicación de alertas, rechazo de modos no implementados, timeout y conservación de logs tras reset. Hay fixtures de color para pruebas unitarias y pruebas de stock con renderizado real; no se confunden entre sí.

El ciclo nominal ejecutado con visor alcanzó `COMPLETE` en **37,04 s simulados**, con **0,22 s de espera del operador simulado**: **36,82 s simulados activos**. No es una promesa de tiempo real ni una medida física de rendimiento. Cada episodio guarda su propio tiempo real y simulado.

En la máquina de desarrollo quedaron los lotes:

- `runs/20260918T231320-a0189432ca/batch.json`: 20 episodios nominales.
- `runs/20260918T231322-fe12d070aa/batch.json`: 18 episodios de escenarios.
- `runs/20260918T231319-06f59f3d36/`: ciclo automático con visor nativo.

Los nombres son UTC; pueden diferir del día local. `runs/` no está versionado. Para compartir evidencias, exportar esas carpetas explícitamente. Cada ejecución nueva imprime su ruta.

### Interpretación de fallos

- `missing`: recipiente retirado de la escena al reset; prueba de ausencia, no stock cero.
- `blocked_return`: se introduce un obstáculo de escenario mientras la caja está en recepción. El sensor de ocupación es ORACLE. La caja se queda en recepción, con reserva e incidente abiertos.
- `pick_timeout`: se congelan los ejes durante la extracción; es inyección de fallo de control, **no un atasco mecánico por contactos**.
- `unknown_stock`: se sustituye la imagen por una imagen negra; es inyección de indisponibilidad visual, **no validación de oclusiones arbitrarias**.
- `empty`: se renderiza un recipiente sin parches de material.

Un fallo esperado tiene `success=false` y `exception_handled_correctly=true`. No se suma como éxito de tarea. El evaluador devuelve código 0 solo si los episodios tuvieron éxito o el fallo esperado correcto, sin falsos éxitos y con observaciones de stock correctas donde existan; devuelve 1 ante regresión. La demo devuelve 1 si el ciclo falla. Los modos no implementados devuelven error de argumentos.

## Convenciones mecánicas y de control

- Metros, segundos, kilogramos; **X** horizontal frente a estantería y recepción; **Z** arriba; **Y positivo** desde el pasillo hacia los recipientes. En transporte Y = −0,48 m; interfaz de los huecos Y = +0,20 m.
- Tuplas `xyz_m` y snapshots: orden **X,Y,Z**. La interfaz `command_axes(x,z,y)` y los actuadores internos usan **X,Z,Y**; la conversión está en el adaptador.
- `PoseObservation` identifica el centro de la cara inferior de la base, no el centro de masa. Base nominal a Z = 0,38 o 0,76 m. No se sobreescriben coordenadas nominales con observaciones.
- Caja: 0,30 × 0,22 × 0,12 m de cuerpo, **más patines de 0,04 m bajo la base**. Huecos de 0,45 m de ancho, apoyos laterales y canal central para horquillas. La recepción usa la misma orientación y tipo de apoyo.
- Todas las dimensiones entregadas a `box()` son completas: la función divide entre dos para los `size` de MuJoCo, que son semiextensiones.
- La horquilla se alinea 18 mm bajo la base y se eleva 55 mm antes de retraer. Con Y extendida solo se permite el movimiento Z explícito de carga/descarga. La regla «no mover X/Z con Y extendida» necesita esta excepción acotada; el transporte horizontal y la elevación de tránsito exigen Y retraída.
- Objetivos de ejes limitados por posición, perfil de consigna de 0,30 m/s y 0,60 m/s², asentamiento <2 mm y <8 mm/s durante cinco ticks, y timeout de movimiento de 18 s simulados. El perfil limita la consigna, no constituye certificación de aceleración física.
- Paso MuJoCo de 2 ms; control cada 10 pasos (50 Hz simulados); visor objetivo 25 Hz reales. El stock se observa por evento, antes y después de la retirada, no en un bucle continuo a 10 Hz.
- La espera humana no consume el timeout de 180 s activos del episodio. El reloj no avanza en `IDLE`. Los tiempos reales separan arranque/inactividad inicial y espera de operador; una pausa manual durante movimiento sigue incluida en `active_wall_s`.

### Qué significa IDEALIZED aquí

Los ejes son articulaciones `slide` actuadas. **Las cajas son cuerpos mocap cinemáticos**, con base, paredes y patines visuales. Durante el acoplamiento se actualiza su posición según los ejes y un offset, y tras el desacoplamiento quedan inmóviles. **Los contactos están desactivados en toda la escena.** Ni los apoyos ni los patines se han validado mecánicamente.

No hay caja libre, peso transportado real, fricción, colisiones verificadas, caída física ni prueba de estabilidad de soporte. La verificación final comprueba pose y velocidad cinemáticas más coherencia logística. `physical_support_verified` y `physical_drops` son `null`, no éxitos ni ceros físicos. No usar esta demo para afirmar recogida robusta o seguridad industrial.

## Stock e integridad

La cámara superior usa una ROI fija calculada con su geometría y el plano nominal de recepción. Dentro de ella se segmentan fondo cian y parches naranjas. Si menos del 85 % del área tiene colores esperados, devuelve `UNKNOWN`. Ocupación naranja <1,5 % se clasifica `EMPTY`, <25 % `LOW`, y el resto `OK`. La calidad hasta 0,95 es una **puntuación heurística, no una probabilidad calibrada**.

La simplificación exige iluminación, colores y plano conocidos. Una oclusión del mismo color que el fondo o el material puede engañar al estimador. No hay reconocimiento de materiales arbitrarios ni conteo de piezas. Los parches solo cambian su visibilidad, no su masa.

La observación posterior es absoluta: no se descuenta consumo otra vez. El inventario conserva observación actual y última observación válida por separado. `UNKNOWN` no pone stock a cero ni resuelve alertas. `LOW/EMPTY` con confianza ≥0,70 abre o actualiza una alerta local por caja; `OK` suficientemente fiable la resuelve. La lógica de cierre por reposición está probada, pero **no hay comando interactivo de reposición** en esta entrega. No se envían emails, compras ni mensajes.

La reserva se mantiene al salir del hueco y ante cualquier fallo; se libera solo tras retorno verificado. No hay recuperación tras reiniciar la aplicación: solo reset completo explícito. Una caja no observada muestra `UNKNOWN`; las observaciones existentes se muestran como históricas con tiempo de simulación y origen.

## Trazabilidad y estructura

Cada ejecución tiene carpeta única con:

- `metadata.json`: modos, seed, escenario, operador, simplificaciones, commit si existe y SHA-256 del código/configuración/lockfile.
- `events.jsonl`: transiciones, acoplamientos, retirada, observaciones y fallos; tiempos reales/simulados, fecha UTC y snapshot del inventario en cada evento.
- `inventory.json`: último snapshot, actualizado mediante reemplazo atómico.
- `stock-before.png` / `stock-after.png`: imágenes RGB de recepción cuando la cámara funciona.
- `result.json`: éxito, motivo, métricas, pose, verificación de entrega y nueva recogida; el evaluador consulta el estado verdadero fuera del controlador.
- `batch.json`: resumen del lote en la carpeta de su primer episodio, si se usa evaluación.

Un único proceso de simulación posee `MjData`. La terminal y el callback del visor solo encolan comandos. El controlador consume snapshots/observaciones y envía acciones; no recorre cuerpos del simulador.

```text
warehouse/
  contracts.py     dataclasses y estados
  scene.py         configuración, MJCF y cámaras
  sim.py           adaptador MuJoCo e IDEALIZED
  motion.py        límites, perfiles y asentamiento
  perception.py    ORACLE para pose; segmentación de stock
  controller.py    estados, guards y confirmación idempotente
  inventory.py     catálogo, reservas y alertas
  persistence.py   eventos, snapshots y versión
  runtime.py       propietario de simulación y coordinación
  demo.py          terminal y visor opcional
  evaluate.py      verdad de simulación y lotes
config/            escena y catálogo
tests/             inventario, percepción y ciclo
```

## Revisión del plan y siguiente hito

El orden del plan es adecuado, pero el hito de 120 minutos debe describirse como **integración**, no validación mecánica. La implementación confirma que conviene fijar antes de ampliar:

1. **Punto de recogida y signos de ejes:** base inferior, conversión X/Y/Z y semiextensiones explícitos. La recogida y la descarga exigen movimientos verticales de docking aunque Y esté extendida.
2. **Dos pruebas de render distintas:** visor abierto y cámara operativa no son equivalentes. Ambos se han ejecutado; no se ha hecho una inspección visual automatizada del escritorio por falta de la CLI de Orca operativa.
3. **Tiempos y verdad de éxito:** se corrigió mediante regresiones el timeout durante espera humana, el avance del reloj en IDLE y el tratamiento de offset cero. Se verifica la nueva recogida de recepción, no solo la llegada de los ejes.

Próximo bloqueo real: sustituir el recipiente cinemático por un cuerpo libre, activar contactos selectivos y verificar extracción, depósito, nueva recogida y retorno por soporte físico, sin escribir su pose durante transporte. Después, integrar pose visual y comparar contra coordenadas fijas con las mismas condiciones. No reutilizar las métricas actuales como evidencia de esos modos.

Pendiente respecto al plan completo: CONTACT, localización VISION, reintento de alineación, segundo tamaño, reposición interactiva, cola de solicitudes, reanudación, benchmark visual emparejado/conjunto reservado y grabación de vídeo de respaldo. No se ha creado vídeo ni diapositiva. No hay ROS, web, LLM en runtime, entrenamiento ni integración externa.

## Guion de 90 segundos

- 0–15 s: «Automatizamos buscar, transportar y devolver cajas; el material lo retira una persona. Hoy mostramos ORACLE + IDEALIZED, no recogida física validada».
- 15–40 s: solicitar `TORNILLOS`, mostrar reserva, pose y entrega. El visor muestra la caja desacoplada en recepción.
- 40–60 s: escribir `confirmar`; explicar el cambio visible de contenido y el aviso `LOW` obtenido de la nueva imagen.
- 60–80 s: mostrar la nueva recogida, retorno al hueco y liberación de reserva tras verificar.
- 80–90 s: enseñar resultados registrados y declarar que CONTACT y localización visual siguen pendientes. No atribuir ahorros ni fiabilidad industrial no medidos.

Referencias técnicas: [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html) y [MJCF](https://mujoco.readthedocs.io/en/stable/XMLreference.html).
