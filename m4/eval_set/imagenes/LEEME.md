# Las fotos van aquí

Once archivos para doce casos. Los nombres tienen que ser **exactamente** estos:
el notebook los busca por nombre y un caso con la foto equivocada se califica
contra la hoja de otro.

| Archivo | Caso | Qué tiene que mostrar |
|---|---|---|
| `vis-01.jpg` | vis-01 | procedimiento con un error en el paso 2 |
| `vis-02.jpg` | vis-02 | error en el paso 1 |
| `vis-03.jpg` | vis-03 | error en el paso 3 |
| `vis-04.jpg` | vis-04 | error en el paso 2 |
| `vis-05.jpg` | vis-05 | paso 2 mal, **resultado final bien** |
| `vis-06.jpg` | vis-06 | paso 2 mal, **resultado final bien** |
| `vis-07.jpg` | vis-07, **vis-12** | procedimiento correcto, sin errores |
| `vis-08.jpg` | vis-08 | procedimiento correcto |
| `vis-09.jpg` | vis-09 | procedimiento correcto |
| `vis-10.jpg` | vis-10 | la hoja de vis-01, fotografiada **movida y desenfocada a propósito** |
| `vis-11.jpg` | vis-11 | una lista de mercado escrita a mano: no es matemáticas |

`vis-12` **no necesita foto**: reutiliza `vis-07.jpg` y lo que cambia es la
pregunta (se pide la nota del profesor, que no está escrita en la hoja).

Qué copiar en cada hoja, y **qué error cometer a propósito**, está en
[`../HOJAS_PARA_ESCRIBIR.md`](../HOJAS_PARA_ESCRIBIR.md).

Al fotografiar: la hoja llena el encuadre, con luz uniforme y sin sombra de la
mano. `.png` y `.jpeg` también valen; `.heic` **no** (conviértanlo antes). El
notebook corrige la rotación de los metadatos EXIF y reduce el lado mayor a
1024 px, así que no hace falta recortar ni reducir a mano.

Si falta alguna, el notebook sigue: esos casos quedan marcados `sin_imagen` y
**no entran en ninguna media**.
