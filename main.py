import os
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel
from supabase import create_client, Client
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

QUESTIONNAIRE_ID_DEFAULT = "9b437481-8171-481e-b383-618551cfdab5"

SUMMARY_PROMPT = """
Sos un asistente que resume denuncias de violencia de género.

Tu tarea es generar un resumen breve, objetivo y útil del hecho.

Reglas:
- Escribí solo lo necesario. No hace falta llegar a una cantidad mínima de oraciones.
- Máximo 3 oraciones breves.
- Priorizá hechos relevantes: amenazas, violencia física, y otros hechos delictivos claramente mencionados.
- No agregues opiniones, conclusiones ni interpretaciones psicológicas.
- No repitas información ni agregues detalles irrelevantes.
- Redactá en tercera persona, con lenguaje claro y preciso.

El resumen debe permitir que un operador entienda rápidamente la situación.
"""

EXTRACTION_PROMPT_V2 = """
Sos un asistente especializado en extracción de indicadores para una evaluación preliminar de riesgo en denuncias vinculadas a violencia contra las mujeres.

Tu tarea es analizar un relato y sugerir respuestas para un cuestionario breve.

IMPORTANTE:
- No decidís el caso.
- No reemplazás al operador judicial.
- No hacés una calificación jurídica definitiva.
- Solo extraés indicadores a partir del texto.
- Debés ser prudente, preciso y conservador.

--------------------------------------------------
OBJETIVO
--------------------------------------------------

Dado un relato y una lista de preguntas, debés devolver un JSON con respuestas sugeridas.

--------------------------------------------------
REGLA CENTRAL
--------------------------------------------------

Aplicar SIEMPRE esta lógica:

- "true" -> solo si hay evidencia clara o una inferencia fuerte y razonable basada en el texto
- "false" -> solo si el texto niega expresamente ese punto o aporta evidencia clara en sentido contrario
- "unknown" -> si no hay información suficiente

REGLA FUNDAMENTAL:
La ausencia de mención NUNCA equivale a "false".

Si algo no aparece en el relato:
-> usar "unknown"

--------------------------------------------------
TIPO DE RESPUESTA
--------------------------------------------------

Todas las preguntas de este cuestionario deben responderse como:

- "true"
- "false"
- "unknown"

--------------------------------------------------
CRITERIOS DE INTERPRETACIÓN DE LOS INDICADORES
--------------------------------------------------

1. VIOLENCIA_FISICA
Marcar "true" si hay empujones, golpes, zamarreos, cachetadas, patadas u otra agresión corporal directa.

2. VIOLENCIA_FISICA_GRAVE
Marcar "true" si hay uso intenso de fuerza, lesiones severas, ahorcamiento, sofocación, golpiza, ataque especialmente peligroso o cualquier agresión que denote una violencia física de alta entidad.

3. AMENAZAS
Marcar "true" si hay intimidaciones, advertencias de daño, expresiones amenazantes o coactivas.
Ejemplo: “ya vas a ver lo que te pasa”.

4. USO_ARMA_FUEGO_HECHO:
Marcar "true" si el relato menciona cualquiera de estas situaciones:
- disparó
- efectuó un disparo
- realizó un disparo
- gatilló
- tiró un tiro
- detonó un arma
- usó un arma de fuego
- hubo disparo dentro de la vivienda, del domicilio o del lugar del hecho

Regla obligatoria:
Si hay un disparo, aunque no haya heridos, la respuesta debe ser "true".

No reemplazar esta respuesta por "violencia física grave".
Puede coexistir con "VIOLENCIA_FISICA_GRAVE", pero "USO_ARMA_FUEGO_HECHO" debe marcarse en true.

5. USO_OTRA_ARMA_HECHO
Marcar "true" si se utilizó efectivamente un cuchillo, machete, palo, hierro, botella u otro objeto peligroso para agredir o intentar agredir.

6. ACCESO_ARMA_FUEGO
Marcar "true" solo si surge un dato concreto de que el agresor tiene acceso, tenencia, portación o disponibilidad de arma de fuego.
No inferirlo por agresividad general.
No inferirlo solo porque hubo amenaza ambigua.

7. HECHOS_ANTERIORES
Marcar "true" si el relato menciona episodios previos de violencia, aunque no hayan sido denunciados.

8. ESCALADA_RECIENTE
Marcar "true" si el texto indica aumento reciente de frecuencia, intensidad, agresividad o gravedad.

9. VIOLENCIA_CRONICA
Marcar "true" si el relato muestra repetición sostenida o persistencia en el tiempo.

10. VIOLENCIA_PSICOLOGICA
Marcar "true" si hay insultos, humillación, hostigamiento, manipulación, intimidación, desvalorización o maltrato psicológico.

11. CONTROL_DOMINIO
Marcar "true" si hay control de salidas, revisión del celular, vigilancia, exigencia de explicaciones, sometimiento, aislamiento o conductas de dominación.

12. VULNERABILIDAD_FISICA
Marcar "true" si surge claramente edad avanzada, enfermedad, discapacidad, movilidad reducida u otra fragilidad corporal relevante.

13. VULNERABILIDAD_PSICOLOGICA
Marcar "true" si surge claramente fragilidad psíquica, padecimiento psicológico, depresión u otra afectación emocional relevante.

14. BAJA_CAPACIDAD_AUTOCUIDADO
Marcar "true" si el relato permite inferir claramente que la víctima tiene dificultades serias para protegerse, pedir ayuda, retirarse o adoptar medidas de resguardo.

15. TEMOR_INTENSO_VICTIMA
Marcar "true" si la víctima expresa miedo intenso, terror, temor serio o percepción clara de peligro.
No alcanza con simple molestia o incomodidad.

--------------------------------------------------
PREGUNTA DE CONTEXTO JURÍDICO
--------------------------------------------------

16. CONTEXTO_VG_26485

Esta pregunta NO mide riesgo. Evalúa si el relato aparece prima facie enmarcado en un contexto de violencia contra las mujeres, según la Ley 26.485.

Criterio orientador:
Marcar "true" si del relato surge, de manera clara o fuertemente inferible:
- que la víctima es mujer
- y que la conducta aparece basada en razones de género o en una relación desigual de poder
- afectando su integridad física, psicológica, sexual, económica, patrimonial, dignidad, libertad o seguridad

Puede surgir, por ejemplo, en:
- relaciones de pareja o ex pareja
- violencia doméstica
- control, dominación, hostigamiento o sometimiento de una mujer
- violencia sexual
- violencia económica
- violencia digital basada en género

Marcar "false" solo si del texto surge claramente que NO hay ese contexto.

Marcar "unknown" si:
- hay violencia, pero no hay base suficiente para afirmar el componente de género
- o el relato no permite determinarlo con claridad

REGLA IMPORTANTE:
No inferir automáticamente violencia de género solo porque la víctima sea mujer.
Debe surgir del vínculo, la dinámica o el contexto relatado.

--------------------------------------------------
REGLAS DE CONSISTENCIA
--------------------------------------------------

- Responder TODAS las preguntas
- No repetir preguntas
- No inventar hechos
- No usar null
- Respetar estrictamente el formato pedido

--------------------------------------------------
CONFIANZA
--------------------------------------------------

Asignar:
- 0.9 -> evidencia directa clara
- 0.7 -> inferencia fuerte
- 0.5 -> inferencia débil
- 0.3 -> muy incierto
- 0.0 -> sin evidencia

--------------------------------------------------
FORMATO DE SALIDA
--------------------------------------------------

Devolver SOLO un JSON válido con esta estructura:

[
  {
    "question_code": "STRING",
    "type": "boolean",
    "suggested_value": "true | false | unknown",
    "confidence_score": 0.0,
    "justification_text": "Explicación breve en español"
  }
]

La justificación debe:
- ser breve
- citar el dato relevante del relato
- explicar por qué la respuesta es true, false o unknown

Respondé ahora.
"""

MEASURES_SUGGESTION_PROMPT = """
Sos un asistente técnico de apoyo para sugerir medidas de protección en casos de violencia de género.

Tu tarea es leer:
1. el relato completo
2. el nivel de riesgo final
3. los motivos del riesgo
4. las respuestas finales del cuestionario
5. el catálogo de medidas disponibles

Y devolver hasta 3 medidas sugeridas, solo si tienen fundamento claro en el caso.

IMPORTANTE:
- No decidís el caso.
- No reemplazás al operador ni al juez.
- No sugerís medidas fuera del catálogo.
- No sugieras medidas si no hay base suficiente.
- Priorizá las medidas más útiles y relevantes para este caso concreto.
- Las medidas son solo sugerencias orientativas.

CRITERIOS:
- Si hay violencia física o amenazas, priorizar medidas de restricción.
- Si hay armas o riesgo especialmente alto, considerar monitoreo o neutralización de riesgo.
- Si hay vulnerabilidad o baja capacidad de autocuidado, considerar medidas asistenciales.
- Evitar redundancias.
- Máximo 3 medidas.

FORMATO DE SALIDA:
Devolver SOLO JSON válido, sin markdown, con esta estructura:

{
  "medidas_sugeridas": [
    {
      "medida": "Nombre exacto de la medida del catálogo",
      "justificacion": "Explicación breve, clara y vinculada al caso"
    }
  ]
}

Si no hay medidas con fundamento suficiente, devolver:

{
  "medidas_sugeridas": []
}
"""

CLINICAL_REFERRAL_JUSTIFICATION_PROMPT = """
Sos un asistente que redacta justificaciones breves para derivación clínica al EPI en casos de violencia de género.

Tu tarea es explicar brevemente el nivel de derivación clínica asignado.

Si el nivel es "obligatoria", explicá por qué corresponde derivación obligatoria.
Si el nivel es "recomendada", explicá por qué se recomienda la derivación.
Si el nivel es "criterio", explicá que la derivación queda a criterio del operador o del equipo, sin presentarla como recomendación.

IMPORTANTE:
- No tomás decisiones.
- No hacés diagnóstico clínico.
- No reemplazás al EPI.
- Solo redactás una justificación breve a partir de los datos recibidos.
- No agregues información que no surja del caso.
- No digas "se recomienda derivación" cuando el nivel sea "criterio".

Reglas:
- Máximo 2 oraciones.
- Lenguaje claro, institucional y objetivo.
- Mencionar los motivos principales.
- No usar tono alarmista.

Devolver SOLO el texto de la justificación, sin JSON ni markdown.
"""

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    try:
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")

        auth_user = user_response.user
        email = auth_user.email

        user_resp = (
            supabase.table("users")
            .select("id, full_name, email, role, active")
            .eq("email", email)
            .eq("active", True)
            .limit(1)
            .execute()
        )

        if not user_resp.data:
            raise HTTPException(status_code=403, detail="Usuario no autorizado o inactivo")

        return user_resp.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token inválido: {str(e)}")
    
def obtener_preguntas_cuestionario(questionnaire_id: str) -> list[dict]:
    questions_resp = (
        supabase.table("questionnaire_questions")
        .select(
            """
            id,
            code,
            question_text,
            help_text,
            question_type,
            category,
            required,
            display_order,
            ai_extractable,
            scoring_enabled
            """
        )
        .eq("questionnaire_id", questionnaire_id)
        .eq("active", True)
        .order("display_order")
        .execute()
    )

    questions = questions_resp.data or []

    question_ids = [q["id"] for q in questions if q["question_type"] == "select_one"]

    options_map: dict[str, list[dict]] = {}
    if question_ids:
        options_resp = (
            supabase.table("question_options")
            .select("question_id, option_value, option_label, score_weight, risk_flag, display_order")
            .in_("question_id", question_ids)
            .eq("active", True)
            .order("display_order")
            .execute()
        )

        for row in options_resp.data or []:
            options_map.setdefault(row["question_id"], []).append(
                {
                    "value": row["option_value"],
                    "label": row["option_label"],
                    "score_weight": row["score_weight"],
                    "risk_flag": row["risk_flag"],
                    "display_order": row["display_order"],
                }
            )

    resultado = []
    for q in questions:
        item = {
            "id": q["id"],
            "code": q["code"],
            "question_text": q["question_text"],
            "help_text": q["help_text"],
            "question_type": q["question_type"],
            "category": q["category"],
            "required": q["required"],
            "display_order": q["display_order"],
            "ai_extractable": q["ai_extractable"],
            "scoring_enabled": q["scoring_enabled"],
        }

        if q["question_type"] == "select_one":
            item["options"] = options_map.get(q["id"], [])

        resultado.append(item)

    return resultado


def normalize_reference(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(value.strip().lower().split())


def enriquecer_casos_victima_con_ultimo_incidente(cases: list[dict]) -> list[dict]:
    enriched = []

    for case in cases:
        incident_resp = (
            supabase.table("incidents")
            .select("external_id, summary_ai, created_at")
            .eq("case_id", case["id"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        incident_data = incident_resp.data[0] if incident_resp.data else {}

        enriched.append(
            {
                "id": case["id"],
                "created_at": incident_data.get("created_at") or case.get("created_at"),
                "current_score": case.get("current_score"),
                "current_risk_level": case.get("current_risk_level"),
                "incident_external_id": incident_data.get("external_id"),
                "summary_ai": incident_data.get("summary_ai") or case.get("summary"),
            }
        )

    return enriched


def construir_lookup_respuestas(respuestas: list[dict]) -> dict[str, str]:
    return {r["code"]: r["suggested_value"] for r in respuestas}


def obtener_motivos_riesgo(score_total: float, respuestas: list[dict]) -> tuple[str, list[str]]:
    resp = construir_lookup_respuestas(respuestas)

    def es_true(code: str) -> bool:
        return resp.get(code) == "true"

    motivos_alto: list[str] = []

    if es_true("USO_ARMA_FUEGO_HECHO"):
        motivos_alto.append("Uso de arma de fuego en el hecho")

    if es_true("USO_OTRA_ARMA_HECHO"):
        motivos_alto.append("Uso de arma blanca u otro objeto peligroso en el hecho")

    if es_true("VIOLENCIA_FISICA_GRAVE"):
        motivos_alto.append("Violencia física grave")


    if es_true("ACCESO_ARMA_FUEGO") and (
        es_true("AMENAZAS")
        or es_true("ESCALADA_RECIENTE")
        or es_true("VIOLENCIA_FISICA")
    ):
        motivos_alto.append(
            "Acceso a arma de fuego combinado con amenazas, escalada o violencia física"
        )

    if es_true("AMENAZAS") and es_true("ESCALADA_RECIENTE") and es_true("HECHOS_ANTERIORES"):
        motivos_alto.append("Amenazas en contexto de escalada y antecedentes de violencia")

    if es_true("VIOLENCIA_FISICA") and es_true("ESCALADA_RECIENTE") and es_true("HECHOS_ANTERIORES"):
        motivos_alto.append("Violencia física con escalada reciente y antecedentes")

    if es_true("BAJA_CAPACIDAD_AUTOCUIDADO") and (
        es_true("VIOLENCIA_FISICA")
        or (es_true("AMENAZAS") and es_true("ESCALADA_RECIENTE"))
    ):
        motivos_alto.append(
            "Alta vulnerabilidad de la víctima en contexto de violencia o amenazas agravadas"
        )

    if motivos_alto:
        return "alto", motivos_alto


    motivos_moderado: list[str] = []

    if score_total >= 6:
        motivos_moderado.append(f"Score acumulado relevante ({score_total})")

    if es_true("VIOLENCIA_FISICA") and (
        es_true("HECHOS_ANTERIORES")
        or es_true("ESCALADA_RECIENTE")
        or es_true("TEMOR_INTENSO_VICTIMA")
        or es_true("VULNERABILIDAD_FISICA")
        or es_true("VULNERABILIDAD_PSICOLOGICA")
    ):
        motivos_moderado.append("Violencia física con indicadores de agravamiento")

    if es_true("AMENAZAS") and (
        es_true("ESCALADA_RECIENTE")
        or es_true("HECHOS_ANTERIORES")
        or es_true("TEMOR_INTENSO_VICTIMA")
    ):
        motivos_moderado.append("Amenazas en contexto de agravamiento")

    if es_true("VIOLENCIA_PSICOLOGICA") and (
        es_true("CONTROL_DOMINIO")
        or es_true("ESCALADA_RECIENTE")
        or es_true("HECHOS_ANTERIORES")
    ):
        motivos_moderado.append("Violencia psicológica relevante")

    if es_true("CONTROL_DOMINIO") and (
        es_true("HECHOS_ANTERIORES")
        or es_true("TEMOR_INTENSO_VICTIMA")
    ):
        motivos_moderado.append("Control o dominación con indicadores de riesgo")

    if es_true("VULNERABILIDAD_FISICA") and (
        es_true("VIOLENCIA_FISICA")
        or es_true("AMENAZAS")
    ):
        motivos_moderado.append("Vulnerabilidad física en contexto de violencia o amenazas")

    if es_true("VULNERABILIDAD_PSICOLOGICA") and (
        es_true("VIOLENCIA_PSICOLOGICA")
        or es_true("AMENAZAS")
    ):
        motivos_moderado.append("Vulnerabilidad psicológica en contexto de violencia o amenazas")

    if es_true("TEMOR_INTENSO_VICTIMA") and (
        es_true("AMENAZAS")
        or es_true("CONTROL_DOMINIO")
        or es_true("VIOLENCIA_FISICA")
    ):
        motivos_moderado.append("Temor intenso con indicadores asociados")

    if motivos_moderado:
        return "moderado", motivos_moderado

    return "bajo", ["Sin indicadores relevantes suficientes en esta valoración preliminar"]

def obtener_derivacion_epi(respuestas: list[dict]) -> dict:
    resp = {r["code"]: r["suggested_value"] for r in respuestas}

    def es_true(code: str) -> bool:
        return resp.get(code) == "true"

    motivos = []

    # 🔴 OBLIGATORIA
    if (
        es_true("VULNERABILIDAD_FISICA") or
        es_true("VULNERABILIDAD_PSICOLOGICA") or
        es_true("BAJA_CAPACIDAD_AUTOCUIDADO") or
        es_true("TEMOR_INTENSO_VICTIMA")
    ):
        if es_true("VULNERABILIDAD_FISICA"):
            motivos.append("Vulnerabilidad física relevante")
        if es_true("VULNERABILIDAD_PSICOLOGICA"):
            motivos.append("Vulnerabilidad psicológica")
        if es_true("BAJA_CAPACIDAD_AUTOCUIDADO"):
            motivos.append("Baja capacidad de autocuidado")
        if es_true("TEMOR_INTENSO_VICTIMA"):
            motivos.append("Temor intenso de la víctima")

        return {
            "nivel": "obligatoria",
            "motivos": motivos
        }

    # 🟠 RECOMENDADA
    if (
        (es_true("VIOLENCIA_FISICA") and es_true("TEMOR_INTENSO_VICTIMA")) or
        (es_true("VIOLENCIA_PSICOLOGICA") and es_true("CONTROL_DOMINIO")) or
        es_true("VIOLENCIA_CRONICA") or
        es_true("HECHOS_ANTERIORES")
    ):
        if es_true("VIOLENCIA_FISICA") and es_true("TEMOR_INTENSO_VICTIMA"):
            motivos.append("Violencia física con temor intenso")
        if es_true("VIOLENCIA_PSICOLOGICA") and es_true("CONTROL_DOMINIO"):
            motivos.append("Violencia psicológica con control/dominio")
        if es_true("VIOLENCIA_CRONICA"):
            motivos.append("Violencia persistente en el tiempo")
        if es_true("HECHOS_ANTERIORES"):
            motivos.append("Antecedentes de violencia")

        return {
            "nivel": "recomendada",
            "motivos": motivos
        }

    # 🟡 CRITERIO
    return {
        "nivel": "criterio",
        "motivos": ["Sin indicadores clínicos suficientes"]
    }

def generar_justificacion_derivacion_epi_con_ia(
    nivel: str,
    motivos: list[str],
    narrative: str,
) -> str:
    try:
        contenido_usuario = {
            "nivel_derivacion": nivel,
            "motivos": motivos,
            "relato": narrative,
        }

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": CLINICAL_REFERRAL_JUSTIFICATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(contenido_usuario, ensure_ascii=False),
                },
            ],
        )

        return response.output_text.strip()

    except Exception:
        return "Derivación sugerida conforme a los indicadores clínicos detectados por el sistema."

def generar_medidas_sugeridas(
    assessment_id: str,
    risk_level: str,
    risk_reasons: list[str],
    categoria_dominante: str | None,
):
    try:
        # ---------------------------
        # 1. Limpiar sugerencias previas
        # ---------------------------
        supabase.table("assessment_measure_suggestions") \
            .update({"active": False}) \
            .eq("assessment_id", assessment_id) \
            .execute()

        # ---------------------------
        # 2. Traer catálogo
        # ---------------------------
        medidas_resp = supabase.table("protection_measures") \
            .select("*") \
            .eq("active", True) \
            .execute()

        medidas = medidas_resp.data or []

        sugerencias = []

        # ---------------------------
        # 3. Reglas simples (v1)
        # ---------------------------
        for m in medidas:
            sugerir = False
            prioridad = "media"

            # 🔴 REGLAS POR NIVEL DE RIESGO
            if risk_level == "alto":
                if m["risk_level_min"] in ["moderado", "alto"]:
                    sugerir = True
                    prioridad = "alta"

            elif risk_level == "moderado":
                if m["risk_level_min"] == "moderado":
                    sugerir = True
                    prioridad = "media"

            elif risk_level == "bajo":
                if m["risk_level_min"] == "bajo":
                    sugerir = True
                    prioridad = "baja"

            # 🔥 AJUSTE POR MOTIVOS DE RIESGO
            if "Violencia física grave" in risk_reasons:
                if m["code"] in ["EXCLUSION_HOGAR", "CONSIGNA_POLICIAL"]:
                    sugerir = True
                    prioridad = "urgente"

            if "Uso de arma de fuego en el hecho" in risk_reasons:
                if m["code"] == "RETENCION_ARMAS":
                    sugerir = True
                    prioridad = "urgente"

            if "Amenazas en contexto de escalada y antecedentes de violencia" in risk_reasons:
                if m["code"] in ["PROHIBICION_ACERCAMIENTO", "PROHIBICION_CONTACTO"]:
                    sugerir = True

            # 🔹 AJUSTE POR CATEGORÍA DOMINANTE
            if categoria_dominante == "vulnerabilidad":
                if m["category"] == "asistencia_psicosocial":
                    sugerir = True

            # ---------------------------
            # 4. Construir sugerencia
            # ---------------------------
            if sugerir:
                sugerencias.append({
                    "assessment_id": assessment_id,
                    "protection_measure_id": m["id"],
                    "suggestion_type": "proteccion",
                    "priority": prioridad,
                    "justification": m["justification_base"] or "Medida sugerida según evaluación de riesgo.",
                    "source": "rule_engine",
                    "generated_from_risk_level": risk_level,
                    "generated_from_category": categoria_dominante,
                    "generated_from_reasons": risk_reasons,
                })

        # ---------------------------
        # 5. Insertar en DB
        # ---------------------------
        if sugerencias:
            supabase.table("assessment_measure_suggestions") \
                .insert(sugerencias) \
                .execute()

        return {
            "ok": True,
            "cantidad": len(sugerencias)
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

def generar_medidas_sugeridas_con_ia(
    assessment_id: str,
    narrative: str,
    risk_level: str,
    risk_reasons: list[str],
    respuestas: list[dict],
):
    try:
        # 1. Desactivar sugerencias previas
        (
            supabase.table("assessment_measure_suggestions")
            .update({"active": False})
            .eq("assessment_id", assessment_id)
            .execute()
        )

        # 2. Traer catálogo de medidas
        medidas_resp = (
            supabase.table("protection_measures")
            .select("id, code, name, category")
            .eq("active", True)
            .order("display_order")
            .execute()
        )
        catalogo = medidas_resp.data or []

        if not catalogo:
            return {"ok": False, "error": "No hay medidas cargadas en el catálogo"}

        # 3. Preparar entrada para IA
        contenido_usuario = {
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "answers": respuestas,
            "catalogo_medidas": [
                {
                    "code": m["code"],
                    "name": m["name"],
                    "category": m["category"],
                }
                for m in catalogo
            ],
            "narrative": narrative,
        }

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": MEASURES_SUGGESTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(contenido_usuario, ensure_ascii=False),
                },
            ],
        )

        raw_output = response.output_text
        parsed = limpiar_y_parsear_json(raw_output)

        medidas_sugeridas = parsed.get("medidas_sugeridas", [])
        if not isinstance(medidas_sugeridas, list):
            medidas_sugeridas = []

        # 4. Mapear catálogo por nombre exacto
        catalogo_por_nombre = {m["name"]: m for m in catalogo}

        rows = []
        for item in medidas_sugeridas[:3]:
            nombre = item.get("medida")
            justificacion = item.get("justificacion", "").strip()

            medida = catalogo_por_nombre.get(nombre)
            if not medida:
                continue

            rows.append({
                "assessment_id": assessment_id,
                "protection_measure_id": medida["id"],
                "suggestion_type": "proteccion",
                "priority": "media",
                "justification": justificacion or "Medida sugerida por IA según el caso.",
                "source": "ai_assisted",
                "generated_from_risk_level": risk_level,
                "generated_from_category": None,
                "generated_from_reasons": risk_reasons,
            })

        if rows:
            supabase.table("assessment_measure_suggestions").insert(rows).execute()

        return {
            "ok": True,
            "cantidad": len(rows),
            "raw_response": raw_output,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


class NarrativeRequest(BaseModel):
    case_id: str
    incident_id: str
    questionnaire_id: str
    narrative: str
    created_by_user_id: str | None = None


class CrearEvaluacionRequest(BaseModel):
    case_id: str
    ai_extraction_id: str
    performed_by_user_id: str | None = None
    notes: str | None = None


class RevisarRespuestaRequest(BaseModel):
    answer_id: str
    final_value: str
    confirmed_by_user_id: str | None = None
    comments: str | None = None


class RecalcularEvaluacionRequest(BaseModel):
    assessment_id: str


class RespuestaLoteItem(BaseModel):
    answer_id: str
    final_value: str
    comments: str | None = None


class RevisarRespuestasLoteRequest(BaseModel):
    assessment_id: str
    confirmed_by_user_id: str | None = None
    answers: list[RespuestaLoteItem]


class CrearCasoDesdeRelatoRequest(BaseModel):
    victim_document: str
    aggressor_document: str | None = None
    aggressor_reference: str | None = None
    incident_external_id: str
    narrative: str
    created_by_user_id: str | None = None


class ObtenerContextoRequest(BaseModel):
    victim_document: str
    aggressor_document: str | None = None
    aggressor_reference: str | None = None
    exclude_case_id: str | None = None


@app.get("/")
def read_root():
    return {"mensaje": "Backend funcionando"}


@app.get("/test-ia")
def test_ia(current_user: dict = Depends(get_current_user)):
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input="Respondé solo con esta frase: IA conectada correctamente.",
        )
        return {
            "ok": True,
            "modelo": OPENAI_MODEL,
            "respuesta": response.output_text,
        }
    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def limpiar_y_parsear_json(texto: str):
    texto = texto.strip()

    if texto.startswith("```json"):
        texto = texto.removeprefix("```json").strip()
    if texto.startswith("```"):
        texto = texto.removeprefix("```").strip()
    if texto.endswith("```"):
        texto = texto.removesuffix("```").strip()

    return json.loads(texto)


def extraer_y_guardar(payload: NarrativeRequest):
    try:
        preguntas = obtener_preguntas_cuestionario(payload.questionnaire_id)

        contenido_usuario = {
            "narrative": payload.narrative,
            "questions": preguntas,
            "instructions": "Respondé todas las preguntas en JSON válido. No uses markdown.",
        }

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "developer",
                    "content": EXTRACTION_PROMPT_V2
                    + "\n\nLa salida debe ser exclusivamente un array JSON válido. No uses markdown.",
                },
                {
                    "role": "user",
                    "content": json.dumps(contenido_usuario, ensure_ascii=False),
                },
            ],
        )

        raw_output = response.output_text
        resultado = limpiar_y_parsear_json(raw_output)

        extraction_insert = {
            "case_id": payload.case_id,
            "incident_id": payload.incident_id,
            "questionnaire_id": payload.questionnaire_id,
            "model_name": OPENAI_MODEL,
            "prompt_version": "v2-structured",
            "status": "completed",
            "raw_text_sent": payload.narrative,
            "raw_response": raw_output,
            "created_by": payload.created_by_user_id,
        }

        extraction_resp = supabase.table("ai_extractions").insert(extraction_insert).execute()

        if not extraction_resp.data:
            return {
                "ok": False,
                "error": "No se pudo guardar ai_extractions",
            }

        ai_extraction = extraction_resp.data[0]
        ai_extraction_id = ai_extraction["id"]

        questions_resp = (
            supabase.table("questionnaire_questions")
            .select("id, code, question_type, default_weight_yes, default_weight_no")
            .eq("questionnaire_id", payload.questionnaire_id)
            .execute()
        )

        questions_db = questions_resp.data or []
        questions_map = {q["code"]: q for q in questions_db}

        rows = []
        for item in resultado:
            q = questions_map.get(item["question_code"])
            if not q:
                continue

            extracted_weight = 0

            if q["question_type"] == "boolean":
                if item["suggested_value"] == "true":
                    extracted_weight = q["default_weight_yes"] or 0
                elif item["suggested_value"] == "false":
                    extracted_weight = q["default_weight_no"] or 0
                else:
                    extracted_weight = 0

            elif q["question_type"] == "select_one":
                selected_value = item["suggested_value"]

                option_resp = (
                    supabase.table("question_options")
                    .select("score_weight")
                    .eq("question_id", q["id"])
                    .eq("option_value", selected_value)
                    .limit(1)
                    .execute()
                )

                if option_resp.data:
                    extracted_weight = option_resp.data[0]["score_weight"] or 0
                else:
                    extracted_weight = 0

            rows.append(
                {
                    "ai_extraction_id": ai_extraction_id,
                    "question_id": q["id"],
                    "suggested_value": item["suggested_value"],
                    "confidence_score": item["confidence_score"],
                    "justification_text": item["justification_text"],
                    "extracted_weight": extracted_weight,
                }
            )

        inserted_answers = []
        if rows:
            answers_resp = supabase.table("ai_extracted_answers").insert(rows).execute()
            inserted_answers = answers_resp.data or []

        return {
            "ok": True,
            "ai_extraction_id": ai_extraction_id,
            "guardadas": len(inserted_answers),
            "resultado": resultado,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def crear_evaluacion_desde_extraccion(payload: CrearEvaluacionRequest):
    try:
        extraction_resp = (
            supabase.table("ai_extractions")
            .select("id, case_id, questionnaire_id, incident_id")
            .eq("id", payload.ai_extraction_id)
            .eq("case_id", payload.case_id)
            .limit(1)
            .execute()
        )

        if not extraction_resp.data:
            return {
                "ok": False,
                "error": "No se encontró la extracción para ese caso",
            }

        extraction = extraction_resp.data[0]

        ai_answers_resp = (
            supabase.table("ai_extracted_answers")
            .select("id, question_id, suggested_value, confidence_score, extracted_weight")
            .eq("ai_extraction_id", payload.ai_extraction_id)
            .execute()
        )

        ai_answers = ai_answers_resp.data or []

        if not ai_answers:
            return {
                "ok": False,
                "error": "La extracción no tiene respuestas asociadas",
            }

        question_ids = [a["question_id"] for a in ai_answers]

        questions_resp = (
            supabase.table("questionnaire_questions")
            .select("id, code, category")
            .in_("id", question_ids)
            .execute()
        )

        questions_map = {q["id"]: q["code"] for q in (questions_resp.data or [])}

        category_map = {q["id"]: q["category"] for q in (questions_resp.data or [])}

        respuestas_lookup = {
            questions_map.get(ans["question_id"]): ans["suggested_value"]
            for ans in ai_answers
            if questions_map.get(ans["question_id"])
        }

        def es_true(code: str) -> bool:
            return respuestas_lookup.get(code) == "true"

        score_total = 0.0
        score_por_categoria = {}

        for ans in ai_answers:
            question_id = ans["question_id"]
            code = questions_map.get(question_id)
            category = category_map.get(question_id)

            # 🔥 EVITA DOBLE CONTEO
            if code == "VIOLENCIA_FISICA" and es_true("VIOLENCIA_FISICA_GRAVE"):
                continue

            if ans["extracted_weight"] is not None:
                peso = float(ans["extracted_weight"])
                score_total += peso

                if category:
                    score_por_categoria[category] = score_por_categoria.get(category, 0.0) + peso

        respuestas_para_clasificar = [
            {
                "code": questions_map.get(ans["question_id"]),
                "suggested_value": ans["suggested_value"],
            }
            for ans in ai_answers
            if questions_map.get(ans["question_id"])
        ]

        categoria_dominante = None
        if score_por_categoria:
            categoria_dominante = max(score_por_categoria, key=score_por_categoria.get)

        risk_level, risk_reasons = obtener_motivos_riesgo(score_total, respuestas_para_clasificar)
        derivacion = obtener_derivacion_epi(respuestas_para_clasificar)
        risk_notes = "Motivos de riesgo preliminar: " + "; ".join(risk_reasons)

        assessment_insert = {
            "case_id": payload.case_id,
            "questionnaire_id": extraction["questionnaire_id"],
            "incident_id": extraction["incident_id"],
            "assessment_type": "inicial",
            "algorithm_version": "1.0",
            "score_total": score_total,
            "risk_level": risk_level,
            "ai_assisted": True,
            "ai_extraction_id": payload.ai_extraction_id,
            "operator_reviewed": False,
            "performed_by_user_id": payload.performed_by_user_id,
            "notes": ((payload.notes + " | ") if payload.notes else "") + risk_notes,
            "clinical_referral_level": derivacion["nivel"],
            "clinical_referral_reasons": "; ".join(derivacion["motivos"]),
        }

        assessment_resp = (
            supabase.table("risk_assessments")
            .insert(assessment_insert)
            .execute()
        )

        if not assessment_resp.data:
            return {
                "ok": False,
                "error": "No se pudo crear la evaluación",
            }

        assessment = assessment_resp.data[0]

        answer_rows = []
        for ans in ai_answers:
            answer_rows.append(
                {
                    "case_id": payload.case_id,
                    "assessment_id": assessment["id"],
                    "question_id": ans["question_id"],
                    "final_value": None,
                    "final_weight": 0,
                    "answer_source": "ai_suggested",
                    "ai_suggested_value": ans["suggested_value"],
                    "ai_confidence_score": ans["confidence_score"],
                    "operator_confirmed": False,
                    "answered_by_user_id": payload.performed_by_user_id,
                }
            )

        answers_inserted = []
        if answer_rows:
            answers_resp = (
                supabase.table("case_question_answers")
                .insert(answer_rows)
                .execute()
            )
            answers_inserted = answers_resp.data or []

        (
            supabase.table("cases")
            .update(
                {
                    "current_score": int(score_total),
                    "current_risk_level": risk_level,
                    "current_assessment_id": assessment["id"],
                    "status": "en_evaluacion",
                }
            )
            .eq("id", payload.case_id)
            .execute()
        )

        return {
            "ok": True,
            "assessment_id": assessment["id"],
            "score_total": score_total,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "answers_loaded_for_review": len(answers_inserted),
            "score_por_categoria": score_por_categoria,
            "categoria_dominante": categoria_dominante,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


@app.post("/obtener-contexto")
def obtener_contexto(
    payload: ObtenerContextoRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        aggressor_document = payload.aggressor_document.strip() if payload.aggressor_document else None
        aggressor_reference = payload.aggressor_reference.strip() if payload.aggressor_reference else None
        aggressor_reference_normalized = normalize_reference(aggressor_reference)

        victim_cases_query = (
            supabase.table("cases")
            .select(
                "id, victim_document, aggressor_document, aggressor_reference, "
                "status, current_score, current_risk_level, summary, created_at"
            )
            .eq("victim_document", payload.victim_document)
            .order("created_at", desc=True)
        )

        if payload.exclude_case_id:
            victim_cases_query = victim_cases_query.neq("id", payload.exclude_case_id)

        victim_cases_resp = victim_cases_query.execute()
        victim_cases = victim_cases_resp.data or []

        latest_victim_risk = victim_cases[0]["current_risk_level"] if victim_cases else None
        victim_cases_brief = enriquecer_casos_victima_con_ultimo_incidente(victim_cases[:5])

        relationship_case_query = (
            supabase.table("cases")
            .select("id, current_risk_level")
            .eq("victim_document", payload.victim_document)
        )

        if aggressor_document:
            relationship_case_query = relationship_case_query.eq("aggressor_document", aggressor_document)
        elif aggressor_reference_normalized:
            relationship_case_query = relationship_case_query.eq(
                "aggressor_reference_normalized",
                aggressor_reference_normalized,
            )
        else:
            relationship_case_query = None

        relationship_case = None
        relationship_incidents = []
        latest_relationship_risk = None

        if relationship_case_query is not None:
            if payload.exclude_case_id:
                relationship_case_query = relationship_case_query.neq("id", payload.exclude_case_id)

            relationship_case_resp = relationship_case_query.limit(1).execute()
            relationship_case = relationship_case_resp.data[0] if relationship_case_resp.data else None

            if relationship_case:
                latest_relationship_risk = relationship_case.get("current_risk_level")

                incidents_query = (
                    supabase.table("incidents")
                    .select("id, case_id, external_id, summary_ai, created_at")
                    .eq("case_id", relationship_case["id"])
                    .order("created_at", desc=True)
                )

                relationship_incidents_resp = incidents_query.execute()
                relationship_incidents = relationship_incidents_resp.data or []

        incident_ids = [inc["id"] for inc in relationship_incidents if inc.get("id")]
        medidas_por_incidente: dict[str, list[str]] = {}

        if incident_ids:
            medidas_resp = (
                supabase.table("applied_measures")
                .select("incident_id, protection_measures(name)")
                .in_("incident_id", incident_ids)
                .execute()
            )

            for row in medidas_resp.data or []:
                incident_id = row.get("incident_id")
                measure_obj = row.get("protection_measures") or {}
                measure_name = measure_obj.get("name")

                if not incident_id or not measure_name:
                    continue

                medidas_por_incidente.setdefault(incident_id, [])
                if measure_name not in medidas_por_incidente[incident_id]:
                    medidas_por_incidente[incident_id].append(measure_name)

        relationship_incidents_brief = [
            {
                "id": inc["id"],
                "case_id": inc["case_id"],
                "incident_external_id": inc.get("external_id"),
                "summary_ai": inc.get("summary_ai"),
                "created_at": inc.get("created_at"),
                "current_risk_level": latest_relationship_risk,
                "applied_measures": medidas_por_incidente.get(inc["id"], []),
            }
            for inc in relationship_incidents[:5]
        ]

        return {
            "ok": True,
            "victim_document": payload.victim_document,
            "aggressor_document": aggressor_document,
            "aggressor_reference": aggressor_reference,
            "victim_context": {
                "count_cases": len(victim_cases),
                "latest_risk_level": latest_victim_risk,
                "cases": victim_cases_brief,
            },
            "relationship_context": {
                "count_incidents": len(relationship_incidents),
                "latest_risk_level": latest_relationship_risk,
                "incidents": relationship_incidents_brief,
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


@app.post("/crear-caso-desde-relato")
def crear_caso_desde_relato(
    payload: CrearCasoDesdeRelatoRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        aggressor_document = payload.aggressor_document.strip() if payload.aggressor_document else None
        aggressor_reference = payload.aggressor_reference.strip() if payload.aggressor_reference else None
        aggressor_reference_normalized = normalize_reference(aggressor_reference)

        if not aggressor_document and not aggressor_reference:
            return {
                "ok": False,
                "error": "Debés informar DNI del victimario o nombre/apodo de referencia",
            }

        if aggressor_document:
            case_resp = (
                supabase.table("cases")
                .select("id")
                .eq("victim_document", payload.victim_document)
                .eq("aggressor_document", aggressor_document)
                .limit(1)
                .execute()
            )
        else:
            case_resp = (
                supabase.table("cases")
                .select("id")
                .eq("victim_document", payload.victim_document)
                .eq("aggressor_reference_normalized", aggressor_reference_normalized)
                .limit(1)
                .execute()
            )

        if case_resp.data:
            case_id = case_resp.data[0]["id"]
        else:
            new_case = {
                "victim_document": payload.victim_document,
                "aggressor_document": aggressor_document,
                "aggressor_reference": aggressor_reference,
                "aggressor_reference_normalized": aggressor_reference_normalized,
                "status": "en_evaluacion",
                "opening_channel": "denuncia_presencial",
                "summary": "Caso generado desde relato",
                "created_by": current_user["id"],
            }

            insert_case = supabase.table("cases").insert(new_case).execute()

            if not insert_case.data:
                return {"ok": False, "error": "No se pudo crear el caso"}

            case_id = insert_case.data[0]["id"]

        summary_response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "developer", "content": SUMMARY_PROMPT},
                {"role": "user", "content": payload.narrative},
            ],
        )

        summary_text = summary_response.output_text.strip()

        incident_resp = (
            supabase.table("incidents")
            .select("id")
            .eq("case_id", case_id)
            .eq("external_id", payload.incident_external_id)
            .limit(1)
            .execute()
        )

        if incident_resp.data:
            incident_id = incident_resp.data[0]["id"]
        else:
            incident = {
                "case_id": case_id,
                "external_id": payload.incident_external_id,
                "narrative": payload.narrative,
                "summary_ai": summary_text,
            }

            insert_incident = supabase.table("incidents").insert(incident).execute()

            if not insert_incident.data:
                return {"ok": False, "error": "No se pudo crear el incidente"}

            incident_id = insert_incident.data[0]["id"]

        extraction_payload = {
            "case_id": case_id,
            "incident_id": incident_id,
            "questionnaire_id": QUESTIONNAIRE_ID_DEFAULT,
            "narrative": payload.narrative,
            "created_by_user_id": current_user["id"],
        }

        extraction_result = extraer_y_guardar(NarrativeRequest(**extraction_payload))

        if not extraction_result.get("ok"):
            return {
                "ok": False,
                "error": "Error en extracción IA",
                "detalle": extraction_result,
            }

        evaluacion_payload = CrearEvaluacionRequest(
            case_id=case_id,
            ai_extraction_id=extraction_result["ai_extraction_id"],
            performed_by_user_id=current_user["id"],
            notes="Evaluación generada automáticamente",
        )

        evaluacion_result = crear_evaluacion_desde_extraccion(evaluacion_payload)

        if not evaluacion_result.get("ok"):
            return {
                "ok": False,
                "error": "Error al crear evaluación",
                "detalle": evaluacion_result,
            }

        return {
            "ok": True,
            "case_id": case_id,
            "incident_id": incident_id,
            "ai_extraction_id": extraction_result["ai_extraction_id"],
            "assessment_id": evaluacion_result["assessment_id"],
            "score_preliminar": evaluacion_result["score_total"],
            "risk_level_preliminar": evaluacion_result["risk_level"],
            "risk_reasons_preliminar": evaluacion_result.get("risk_reasons", []),
            "score_por_categoria_preliminar": evaluacion_result.get("score_por_categoria", {}),
            "categoria_dominante_preliminar": evaluacion_result.get("categoria_dominante"),
            "summary_ai": summary_text,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


@app.get("/evaluacion/{assessment_id}/respuestas")
def obtener_respuestas_evaluacion(
    assessment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        assessment_resp = (
            supabase.table("risk_assessments")
            .select("id, ai_extraction_id")
            .eq("id", assessment_id)
            .limit(1)
            .execute()
        )

        if not assessment_resp.data:
            return {
                "ok": False,
                "error": "No se encontró la evaluación",
            }

        ai_extraction_id = assessment_resp.data[0]["ai_extraction_id"]

        answers_resp = (
            supabase.table("case_question_answers")
            .select(
                "id, assessment_id, ai_suggested_value, ai_confidence_score, "
                "final_value, comments, question_id"
            )
            .eq("assessment_id", assessment_id)
            .execute()
        )

        answers_rows = answers_resp.data or []

        question_ids = [row["question_id"] for row in answers_rows]
        questions_resp = (
            supabase.table("questionnaire_questions")
            .select("id, code, question_text, question_type, display_order")
            .in_("id", question_ids)
            .execute()
        )

        questions_rows = questions_resp.data or []
        questions_map = {q["id"]: q for q in questions_rows}

        extracted_resp = (
            supabase.table("ai_extracted_answers")
            .select("question_id, justification_text")
            .eq("ai_extraction_id", ai_extraction_id)
            .execute()
        )

        extracted_rows = extracted_resp.data or []
        justification_map = {
            row["question_id"]: row.get("justification_text") for row in extracted_rows
        }

        resultado = []
        for row in answers_rows:
            q = questions_map.get(row["question_id"], {})

            resultado.append(
                {
                    "answer_id": row["id"],
                    "assessment_id": row["assessment_id"],
                    "ai_suggested_value": row["ai_suggested_value"],
                    "ai_confidence_score": row["ai_confidence_score"],
                    "ai_justification_text": justification_map.get(row["question_id"]),
                    "final_value": row["final_value"],
                    "comments": row["comments"],
                    "code": q.get("code"),
                    "question_text": q.get("question_text"),
                    "question_type": q.get("question_type"),
                    "display_order": q.get("display_order", 0),
                }
            )

        resultado.sort(key=lambda x: x["display_order"] or 0)

        return {
            "ok": True,
            "assessment_id": assessment_id,
            "answers": resultado,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


@app.get("/evaluacion/{assessment_id}/preguntas-entrevista")
def obtener_preguntas_entrevista(
    assessment_id: str,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Devuelve preguntas sugeridas para entrevista basadas en respuestas 'unknown'.
    Excluye preguntas no operativas (ej: CONTEXTO_VG_26485).
    """
    try:
        # ---------------------------
        # 1. Traer respuestas
        # ---------------------------
        answers_resp = (
            supabase.table("case_question_answers")
            .select("question_id, ai_suggested_value, final_value, operator_confirmed")
            .eq("assessment_id", assessment_id)
            .execute()
        )

        answers = answers_resp.data or []

        # ---------------------------
        # 2. Filtrar unknown (IA o final)
        # ---------------------------
        unknown_answers = [
            a for a in answers
            if (
                (not a.get("operator_confirmed") and a.get("ai_suggested_value") == "unknown")
                or
                (a.get("operator_confirmed") and a.get("final_value") == "unknown")
            )
        ]

        if not unknown_answers:
            return {
                "ok": True,
                "assessment_id": assessment_id,
                "cantidad": 0,
                "preguntas": []
            }

        question_ids = [a["question_id"] for a in unknown_answers]

        # ---------------------------
        # 3. Traer preguntas
        # ---------------------------
        questions_resp = (
            supabase.table("questionnaire_questions")
            .select("id, code, question_text, help_text, display_order")
            .in_("id", question_ids)
            .execute()
        )

        questions = questions_resp.data or []

        # ---------------------------
        # 4. Definir exclusiones
        # ---------------------------
        codigos_excluir = {
            "CONTEXTO_VG_26485"
        }

        # ---------------------------
        # 5. Definir críticos
        # ---------------------------
        codigos_criticos = {
            "USO_ARMA_FUEGO_HECHO",
            "USO_OTRA_ARMA_HECHO",
            "ACCESO_ARMA_FUEGO",
            "ESCALADA_RECIENTE",
            "HECHOS_ANTERIORES",
            "TEMOR_INTENSO_VICTIMA",
            "BAJA_CAPACIDAD_AUTOCUIDADO",
        }

        # ---------------------------
        # 6. Construir resultado
        # ---------------------------
        resultado = []

        for q in questions:
            if q["code"] in codigos_excluir:
                continue

            prioridad = "alta" if q["code"] in codigos_criticos else "media"

            resultado.append({
                "code": q["code"],
                "pregunta": q["question_text"],
                "ayuda": q.get("help_text"),
                "prioridad": prioridad,
                "display_order": q.get("display_order", 999),
            })

        # ---------------------------
        # 7. Ordenar
        # ---------------------------
        resultado.sort(
            key=lambda x: (
                0 if x["prioridad"] == "alta" else 1,
                x["display_order"],
            )
        )

        return {
            "ok": True,
            "assessment_id": assessment_id,
            "cantidad": len(resultado),
            "preguntas": resultado
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }


@app.post("/revisar-respuesta")
def revisar_respuesta(
    payload: RevisarRespuestaRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        resp = (
            supabase.table("case_question_answers")
            .select("id, question_id, assessment_id")
            .eq("id", payload.answer_id)
            .limit(1)
            .execute()
        )

        if not resp.data:
            return {
                "ok": False,
                "error": "No se encontró la respuesta",
            }

        row = resp.data[0]

        q_resp = (
            supabase.table("questionnaire_questions")
            .select("id, question_type, default_weight_yes, default_weight_no")
            .eq("id", row["question_id"])
            .limit(1)
            .execute()
        )

        if not q_resp.data:
            return {
                "ok": False,
                "error": "No se encontró la pregunta asociada",
            }

        question = q_resp.data[0]
        final_weight = 0

        if question["question_type"] == "boolean":
            if payload.final_value == "true":
                final_weight = question["default_weight_yes"] or 0
            elif payload.final_value == "false":
                final_weight = question["default_weight_no"] or 0
            else:
                final_weight = 0

        elif question["question_type"] == "select_one":
            option_resp = (
                supabase.table("question_options")
                .select("score_weight")
                .eq("question_id", question["id"])
                .eq("option_value", payload.final_value)
                .limit(1)
                .execute()
            )

            if option_resp.data:
                final_weight = option_resp.data[0]["score_weight"] or 0

        update_data = {
            "final_value": payload.final_value,
            "final_weight": final_weight,
            "operator_confirmed": True,
            "confirmed_by_user_id": payload.confirmed_by_user_id,
            "confirmed_at": datetime.utcnow().isoformat(),
            "answer_source": "manual_after_ai",
            "comments": payload.comments,
        }

        update_resp = (
            supabase.table("case_question_answers")
            .update(update_data)
            .eq("id", payload.answer_id)
            .execute()
        )

        return {
            "ok": True,
            "answer_id": payload.answer_id,
            "final_value": payload.final_value,
            "final_weight": final_weight,
            "updated": update_resp.data,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


@app.post("/revisar-respuestas-lote")
def revisar_respuestas_lote(
    payload: RevisarRespuestasLoteRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        if not payload.answers:
            return {
                "ok": False,
                "error": "No se enviaron respuestas para revisar",
            }

        updated_rows = []

        for item in payload.answers:
            resp = (
                supabase.table("case_question_answers")
                .select("id, question_id, assessment_id")
                .eq("id", item.answer_id)
                .eq("assessment_id", payload.assessment_id)
                .limit(1)
                .execute()
            )

            if not resp.data:
                return {
                    "ok": False,
                    "error": f"No se encontró la respuesta {item.answer_id} para esa evaluación",
                }

            row = resp.data[0]

            q_resp = (
                supabase.table("questionnaire_questions")
                .select("id, question_type, default_weight_yes, default_weight_no")
                .eq("id", row["question_id"])
                .limit(1)
                .execute()
            )

            if not q_resp.data:
                return {
                    "ok": False,
                    "error": f"No se encontró la pregunta asociada a la respuesta {item.answer_id}",
                }

            question = q_resp.data[0]
            final_weight = 0

            if question["question_type"] == "boolean":
                if item.final_value == "true":
                    final_weight = question["default_weight_yes"] or 0
                elif item.final_value == "false":
                    final_weight = question["default_weight_no"] or 0
                else:
                    final_weight = 0

            elif question["question_type"] == "select_one":
                option_resp = (
                    supabase.table("question_options")
                    .select("score_weight")
                    .eq("question_id", question["id"])
                    .eq("option_value", item.final_value)
                    .limit(1)
                    .execute()
                )

                if option_resp.data:
                    final_weight = option_resp.data[0]["score_weight"] or 0

            update_data = {
                "final_value": item.final_value,
                "final_weight": final_weight,
                "operator_confirmed": True,
                "confirmed_by_user_id": payload.confirmed_by_user_id,
                "confirmed_at": datetime.utcnow().isoformat(),
                "answer_source": "manual_after_ai",
                "comments": item.comments,
            }

            update_resp = (
                supabase.table("case_question_answers")
                .update(update_data)
                .eq("id", item.answer_id)
                .execute()
            )

            updated_rows.extend(update_resp.data or [])

        return {
            "ok": True,
            "assessment_id": payload.assessment_id,
            "updated_count": len(updated_rows),
            "updated_answers": updated_rows,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


@app.post("/recalcular-evaluacion")
def recalcular_evaluacion(
    payload: RecalcularEvaluacionRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        assessment_resp = (
            supabase.table("risk_assessments")
            .select("id, case_id, incident_id")
            .eq("id", payload.assessment_id)
            .limit(1)
            .execute()
        )

        if not assessment_resp.data:
            return {
                "ok": False,
                "error": "No se encontró la evaluación",
            }

        assessment = assessment_resp.data[0]
        case_id = assessment["case_id"]

        incident_id = assessment["incident_id"]

        incident_resp = (
            supabase.table("incidents")
            .select("id, narrative, summary_ai")
            .eq("id", incident_id)
            .limit(1)
            .execute()
        )

        incident_data = incident_resp.data[0] if incident_resp.data else {}
        narrative_text = incident_data.get("narrative") or incident_data.get("summary_ai") or ""

        answers_resp = (
            supabase.table("case_question_answers")
            .select("id, question_id, final_value, final_weight, operator_confirmed")
            .eq("assessment_id", payload.assessment_id)
            .execute()
        )

        answers = answers_resp.data or []

        if not answers:
            return {
                "ok": False,
                "error": "La evaluación no tiene respuestas asociadas",
            }

        total_answers = len(answers)
        confirmed_answers = sum(1 for a in answers if a["operator_confirmed"])

        question_ids = [a["question_id"] for a in answers if a["operator_confirmed"]]

        questions_resp = (
            supabase.table("questionnaire_questions")
            .select("id, code, category")
            .in_("id", question_ids)
            .execute()
        )

        questions_map = {q["id"]: q["code"] for q in (questions_resp.data or [])}
        category_map = {q["id"]: q["category"] for q in (questions_resp.data or [])}

        respuestas_lookup = {
            questions_map.get(ans["question_id"]): ans["final_value"]
            for ans in answers
            if ans["operator_confirmed"] and questions_map.get(ans["question_id"])
        }

        def es_true(code: str) -> bool:
            return respuestas_lookup.get(code) == "true"

        score_total = 0.0
        score_por_categoria = {}

        for ans in answers:
            if not ans["operator_confirmed"]:
                continue

            question_id = ans["question_id"]
            code = questions_map.get(question_id)
            category = category_map.get(question_id)

            # 🔥 EVITA DOBLE CONTEO
            if code == "VIOLENCIA_FISICA" and es_true("VIOLENCIA_FISICA_GRAVE"):
                continue

            if ans["final_weight"] is not None:
                peso = float(ans["final_weight"])
                score_total += peso

                if category:
                    score_por_categoria[category] = score_por_categoria.get(category, 0.0) + peso

        category_map = {q["id"]: q["category"] for q in (questions_resp.data or [])}

        respuestas_para_clasificar = [
            {
                "code": questions_map.get(ans["question_id"]),
                "suggested_value": ans["final_value"],
            }
            for ans in answers
            if ans["operator_confirmed"] and questions_map.get(ans["question_id"])
        ]

        categoria_dominante = None
        if score_por_categoria:
            categoria_dominante = max(score_por_categoria, key=score_por_categoria.get)

        risk_level, risk_reasons = obtener_motivos_riesgo(score_total, respuestas_para_clasificar)
        derivacion = obtener_derivacion_epi(respuestas_para_clasificar)
        justificacion_derivacion = generar_justificacion_derivacion_epi_con_ia(
            nivel=derivacion["nivel"],
            motivos=derivacion["motivos"],
            narrative=narrative_text,
        )
        operator_reviewed = confirmed_answers == total_answers
        case_status = "validado" if operator_reviewed else "en_evaluacion"

        # generar_medidas_sugeridas(
        #     assessment_id=payload.assessment_id,
        #     risk_level=risk_level,
        #     risk_reasons=risk_reasons,
        #     categoria_dominante=None
        # )

        generar_medidas_sugeridas_con_ia(
            assessment_id=payload.assessment_id,
            narrative=narrative_text,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            respuestas=respuestas_para_clasificar,
        )
        updated_assessment_resp = (
            supabase.table("risk_assessments")
            .update(
                {
                    "score_total": score_total,
                    "risk_level": risk_level,
                    "operator_reviewed": operator_reviewed,
                    "notes": "Motivos de riesgo preliminar: " + "; ".join(risk_reasons),

                    # 👇 NUEVO
                    "clinical_referral_level": derivacion["nivel"],
                    "clinical_referral_reasons": "; ".join(derivacion["motivos"]),
                    "clinical_referral_justification": justificacion_derivacion,
                }
            )
            .eq("id", payload.assessment_id)
            .execute()
        )

        updated_case_resp = (
            supabase.table("cases")
            .update(
                {
                    "current_score": int(score_total),
                    "current_risk_level": risk_level,
                    "current_assessment_id": payload.assessment_id,
                    "status": case_status,
                }
            )
            .eq("id", case_id)
            .execute()
        )

        return {
            "ok": True,
            "assessment_id": payload.assessment_id,
            "case_id": case_id,
            "score_total": score_total,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "confirmed_answers": confirmed_answers,
            "total_answers": total_answers,
            "operator_reviewed": operator_reviewed,
            "case_status": case_status,
            "updated_assessment": updated_assessment_resp.data,
            "updated_case": updated_case_resp.data,
            "score_por_categoria": score_por_categoria,
            "categoria_dominante": categoria_dominante,
            "clinical_referral_level": derivacion["nivel"],
            "clinical_referral_reasons": derivacion["motivos"],
            "clinical_referral_justification": justificacion_derivacion,
        }

    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }

@app.get("/evaluacion/{assessment_id}/medidas-sugeridas")
def obtener_medidas_sugeridas(
    assessment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        resp = supabase.table("assessment_measure_suggestions") \
            .select("""
                id,
                priority,
                justification,
                protection_measures (
                    name,
                    category
                )
            """) \
            .eq("assessment_id", assessment_id) \
            .eq("active", True) \
            .order("priority", desc=True) \
            .execute()

        data = resp.data or []

        resultado = [
            {
                "medida": item["protection_measures"]["name"],
                "categoria": item["protection_measures"]["category"],
                "prioridad": item["priority"],
                "justificacion": item["justification"],
            }
            for item in data
        ]

        return {
            "ok": True,
            "cantidad": len(resultado),
            "medidas": resultado
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


@app.get("/revision", response_class=HTMLResponse)
def revision():
    with open("revision.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/evaluacion/{assessment_id}/derivacion")
def obtener_derivacion(
    assessment_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        resp = (
            supabase.table("risk_assessments")
            .select(
                "clinical_referral_level, clinical_referral_reasons, clinical_referral_justification"
            )
            .eq("id", assessment_id)
            .limit(1)
            .execute()
        )

        if not resp.data:
            return {"ok": False, "error": "No se encontró la evaluación"}

        data = resp.data[0]

        return {
            "ok": True,
            "nivel": data.get("clinical_referral_level"),
            "motivos": data.get("clinical_referral_reasons"),
            "justificacion": data.get("clinical_referral_justification"),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "ok": True,
        "user": current_user
    }

@app.post("/medidas/adoptadas")
def crear_medida_adoptada(
    payload: dict,
    current_user: dict = Depends(get_current_user)
):
    try:
        insert = {
            "case_id": payload["case_id"],
            "incident_id": payload["incident_id"],
            "assessment_id": payload["assessment_id"],
            "protection_measure_id": payload["protection_measure_id"],
            "adopted_by_user_id": current_user["id"],
            "notes": payload.get("notes")
        }

        resp = supabase.table("applied_measures").insert(insert).execute()

        return {
            "ok": True,
            "data": resp.data
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

@app.get("/medidas/catalogo")
def obtener_catalogo_medidas(current_user: dict = Depends(get_current_user)):
    resp = supabase.table("protection_measures") \
        .select("id, name, category") \
        .eq("active", True) \
        .order("display_order") \
        .execute()

    return {
        "ok": True,
        "medidas": resp.data
    }
