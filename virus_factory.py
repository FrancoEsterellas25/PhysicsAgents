"""
================================================================================
virus_factory.py  ·  Fabrica de Virus — Catalogo Epidemiologico Calibrado
================================================================================
Catalogo de enfermedades reales calibradas para el motor continuo
(espacio fisico / dinamica de Langevin) del Simulador SEIRS-D en PhysicsAgents.

Parametros del motor que este modulo controla:
    tau_max      : Dosis de tolerancia maxima del agente. Inversa de la
                   contagiosidad. Base = omega_in * tau_max por agente.
    ell          : Radio del kernel gaussiano de dispersion de aerosol (metros).
    delta_ext    : Decaimiento viral en transito / exteriores (dia^-1).
    delta_cerrado: Decaimiento viral en espacios cerrados (dia^-1).
    delta_abierto: Decaimiento viral en espacios abiertos (dia^-1).
    lam          : Pendiente de la funcion logistica de letalidad.
    k_E          : Forma de la distribucion NegBin para incubacion (E).
    p_E          : Prob. de exito de NegBin(k_E, p_E). Media = k_E*(1-p_E)/p_E.
    mu_R         : Dias promedio de inmunidad adquirida (R -> S).
    M_R          : Cap maximo de dias de inmunidad.

Uso:
    from virus_factory import VIRUS_CATALOG, apply_to_simulation
    sim = ContinuousSEIRSDSimulation(N=1600)
    apply_to_simulation(sim, VIRUS_CATALOG["measles"])
    sim.run()

Fuentes:
    Wells-Riley (quanta -> tau_max), van Doremalen 2020 (delta_ext),
    Biggerstaff 2014, Althaus 2014, Feldmann & Geisbert 2011, WHO.
================================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict
from tabulate import tabulate

# Forzar UTF-8 en consola Windows (cp1252 no admite caracteres especiales)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ===========================================================================
# Dataclass del perfil virico
# ===========================================================================

@dataclass(frozen=True)
class VirusProfile:
    """
    Perfil fisico-biologico de un patogeno calibrado para el motor Langevin
    de PhysicsAgents (ContinuousSEIRSDSimulation).

    Attributes
    ----------
    name : str
        Nombre canonico de la enfermedad.
    description : str
        Descripcion epidemiologica breve.
    tau_max : float
        Dosis de tolerancia maxima (unidades internas del modelo).
        Bajo = muy contagioso (basta poca dosis acumulada para infectar).
        Alto = requiere exposicion intensa o prolongada.
    ell : float
        Radio del kernel gaussiano de dispersion de aerosol (metros).
        Grande = aerosoles finos de largo alcance (sarampion, TB).
        Pequenio = gotas pesadas / contacto cercano (ebola).
    delta_ext : float
        Tasa de inactivacion viral en transito/exteriores (dia^-1).
        Alta = virus muere rapido fuera del huesped.
    delta_cerrado : float
        Tasa de inactivacion viral en espacios cerrados (hogar, trabajo).
        Tipicamente menor que delta_ext (menor ventilacion, mayor persistencia).
    delta_abierto : float
        Tasa de inactivacion viral en espacios abiertos (plaza, mercado).
        Tipicamente mayor que delta_ext (UV, viento).
    lam : float
        Pendiente logistica de letalidad:
            P(muerte) = sigmoid(lam * (E_index - omega_ad))
        Mayor lam = mortalidad se dispara bruscamente con la carga viral.
    k_E : int
        Forma de NegBin para el periodo de incubacion.
    p_E : float
        Probabilidad de NegBin(k_E, p_E). Media = k_E*(1-p_E)/p_E dias.
    mu_R : float
        Dias promedio de inmunidad adquirida post-infeccion.
    M_R : float
        Cap maximo de dias de inmunidad (cota distribucional superior).
    r0_ref : str
        R0 de referencia de la literatura (solo informativo).
    cfr_ref : str
        CFR de referencia (solo informativo).
    notes : str
        Notas de calibracion y fuentes bibliograficas.
    """
    name          : str
    description   : str
    tau_max       : float
    ell           : float
    delta_ext     : float
    delta_cerrado : float
    delta_abierto : float
    lam           : float
    k_E           : int
    p_E           : float
    mu_R          : float
    M_R           : float
    r0_ref        : str = ""
    cfr_ref       : str = ""
    notes         : str = ""

    @property
    def mean_incubation_days(self) -> float:
        """Media del periodo de incubacion (dias), segun NegBin(k_E, p_E)."""
        return self.k_E * (1.0 - self.p_E) / self.p_E

    @property
    def as_dict(self) -> Dict[str, Any]:
        """Parametros inyectables al simulador como diccionario."""
        return {
            "tau_max"       : self.tau_max,
            "ell"           : self.ell,
            "delta_ext"     : self.delta_ext,
            "delta_cerrado" : self.delta_cerrado,
            "delta_abierto" : self.delta_abierto,
            "lam"           : self.lam,
            "k_E"           : self.k_E,
            "p_E"           : self.p_E,
            "mu_R"          : self.mu_R,
            "M_R"           : self.M_R,
        }

    def __str__(self) -> str:
        return (
            f"VirusProfile({self.name!r})\n"
            f"  tau_max       = {self.tau_max}   [dosis tolerada]\n"
            f"  ell           = {self.ell} m     [radio aerosol]\n"
            f"  delta_ext     = {self.delta_ext}/dia [transito/exterior]\n"
            f"  delta_cerrado = {self.delta_cerrado}/dia [espacios cerrados]\n"
            f"  delta_abierto = {self.delta_abierto}/dia [espacios abiertos]\n"
            f"  lam           = {self.lam}       [pendiente letalidad]\n"
            f"  NegBin(k={self.k_E}, p={self.p_E}) -> incubacion media "
            f"{self.mean_incubation_days:.1f} dias\n"
            f"  mu_R = {self.mu_R} dias, M_R = {self.M_R} dias\n"
            f"  R0 ref: {self.r0_ref}\n"
            f"  CFR ref: {self.cfr_ref}"
        )


# ===========================================================================
# CATALOGO DE VIRUS
# ===========================================================================

VIRUS_CATALOG: Dict[str, VirusProfile] = {}

# ---------------------------------------------------------------------------
# 1. SARAMPION (Measles morbillivirus)
# ---------------------------------------------------------------------------
# R0 = 12-18. Aerosoles finos, contagio a >10m, inmunidad vitalicia.
# Wells-Riley: quanta ~570/h. delta_ext bajo: viable 2h en el aire.
# tau_max muy bajo: dosis infecciosa minima (1-10 viriones equivalentes).
VIRUS_CATALOG["measles"] = VirusProfile(
    name          = "Sarampion (Measles)",
    description   = (
        "Enfermedad viral de maxima contagiosidad. Aerosoles finos (<5um) "
        "permanecen suspendidos hasta 2h. Inmunidad vitalicia post-infeccion. "
        "Erradicable solo con cobertura vacunal >95% (MMR)."
    ),
    tau_max       = 2.0,     # muy bajo en escala interna (omega_in * tau_max base=20)
    ell           = 4.0,     # metros; aerosoles de largo alcance
    delta_ext     = 0.3,     # /dia en exterior (~2h de vida)
    delta_cerrado = 0.15,    # /dia en cerrado (>2h en espacios sin ventilacion)
    delta_abierto = 1.5,     # /dia en abierto (UV + dispersion)
    lam           = 1.5,     # CFR baja en sanos; alta en <5 anos
    k_E           = 5,
    p_E           = 0.333,   # media E = 5*0.667/0.333 = 10.0 dias
    mu_R          = 36500.0, # ~vitalicio (100 anos)
    M_R           = 36500.0,
    r0_ref        = "R0 = 12-18 (Fine & Clarkson, 1982)",
    cfr_ref       = "CFR ~0.1-1% (paises desarrollados); hasta 10% (recursos limitados)",
    notes         = (
        "tau_max=2.0 en escala interna: con omega_in~0.5, tau_infection~1.0 "
        "(baja dosis infecciosa). ell=4m: contagio documentado a >10m "
        "(Remington et al. 1985). delta_cerrado=0.15: viable >2h en aulas "
        "sin ventilacion (Barker et al. 2013)."
    ),
)

# ---------------------------------------------------------------------------
# 2. COVID-19 DELTA (SARS-CoV-2 B.1.617.2)
# ---------------------------------------------------------------------------
# R0 = 5-6. Aerosoles mixtos (finos + gotas). Waning inmunidad 4-6 meses.
# van Doremalen 2020: viable ~3h en aerosol. Miller 2021: quanta ~120/h.
VIRUS_CATALOG["covid19_delta"] = VirusProfile(
    name          = "COVID-19 Delta (SARS-CoV-2 B.1.617.2)",
    description   = (
        "Variante Delta del SARS-CoV-2. Transmision aerea predominante "
        "(aerosoles finos + gotas medianas). Super-spreading en interiores "
        "mal ventilados. Inmunidad post-infeccion con waning acelerado."
    ),
    tau_max       = 6.0,     # contagiosidad media-alta en escala interna
    ell           = 2.5,     # metros; alcance intermedio
    delta_ext     = 1.5,     # /dia exterior (~3h viabilidad)
    delta_cerrado = 0.5,     # /dia cerrado (persistencia en aire estancado)
    delta_abierto = 4.0,     # /dia abierto (rapida disipacion)
    lam           = 3.5,     # CFR ~1-2% sin vacuna
    k_E           = 5,
    p_E           = 0.50,    # media E = 5*0.5/0.5 = 5.0 dias
    mu_R          = 120.0,   # ~4 meses
    M_R           = 180.0,   # cap 6 meses
    r0_ref        = "R0 = 5-6 (Fisman & Tuite, 2021; Li et al. 2021)",
    cfr_ref       = "CFR ~1-2% sin vacunacion (WHO 2021)",
    notes         = (
        "tau_max=6.0: requiere acumulacion moderada de dosis. "
        "delta_cerrado=0.5: virus persiste horas en interiores "
        "(van Doremalen et al. 2020). mu_R corto por waning "
        "demostrado a los 3-6 meses (Goldberg et al. 2021)."
    ),
)

# ---------------------------------------------------------------------------
# 3. INFLUENZA ESTACIONAL (H3N2 / H1N1)
# ---------------------------------------------------------------------------
# R0 = 1.2-1.4. Transmision por gotas grandes + contacto. CFR ~0.1%.
# Harper 1961: T1/2 ~30min en aire seco. Deriva antigenica rapida.
VIRUS_CATALOG["influenza"] = VirusProfile(
    name          = "Influenza Estacional (H3N2/H1N1)",
    description   = (
        "Gripe estacional causada por Influenza A. Transmision por gotas "
        "grandes y contacto directo; aerosoles secundarios en interiores. "
        "Deriva antigenica anual obliga a reformulacion de vacuna."
    ),
    tau_max       = 15.0,    # requiere dosis considerablemente alta
    ell           = 1.2,     # metros; gotas pesadas de corto alcance
    delta_ext     = 4.0,     # /dia exterior (T1/2 ~2.5h)
    delta_cerrado = 2.0,     # /dia cerrado (algo mas estable que exterior)
    delta_abierto = 8.0,     # /dia abierto (UV + viento lo inactivan rapido)
    lam           = 0.8,     # CFR muy baja en poblacion general
    k_E           = 2,
    p_E           = 0.50,    # media E = 2*0.5/0.5 = 2.0 dias
    mu_R          = 150.0,   # ~5 meses (temporada, waning rapido)
    M_R           = 210.0,
    r0_ref        = "R0 = 1.2-1.4 (Biggerstaff et al. 2014)",
    cfr_ref       = "CFR ~0.1% general; >0.5% en >=65 anos (CDC 2020)",
    notes         = (
        "tau_max=15.0: dosis infecciosa mas alta que SARS-CoV-2. "
        "delta_cerrado=2.0 basado en Harper (1961): T1/2 ~25min a RH=50%. "
        "mu_R corto por waning funcional (deriva antigenica)."
    ),
)

# ---------------------------------------------------------------------------
# 4. H1N1 (Gripe Pandémica 2009)
# ---------------------------------------------------------------------------
# R0 = 1.4-1.6. Transmisión por gotas respiratorias.
# Impacto demográfico sesgado hacia población joven (inmunidad cruzada previa en ancianos).
# Modelo: Curva de crecimiento acelerado con caída por agotamiento de susceptibles.
VIRUS_CATALOG["h1n1_2009"] = VirusProfile(
    name            = "Gripe Pandémica 2009",
    description     = "Variante de influenza con alta tasa de ataque en jóvenes. Transmisión por gotas y contacto.",
    tau_max         = 7.0,
    ell             = 1.5,     # Gotas de corto alcance
    delta_ext       = 2.0,     # Moderada estabilidad ambiental
    delta_cerrado   = 1.0, 
    delta_abierto   = 3.0, 
    lam             = 0.2,     # CFR moderada-baja
    k_E             = 2,       # Incubación rápida (1-4 días)
    p_E             = 0.5, 
    mu_R            = 365.0,   # Inmunidad duradera (~1 año o más)
    M_R             = 365.0,
    r0_ref          = "R0 ~ 1.4-1.6 (CDC, 2009)",
    cfr_ref         = "CFR ~ 0.02%",
    notes           = "Curva epidémica rápida característica de influenza."
)

# ---------------------------------------------------------------------------
# 5. EBOLA (Ebolavirus cepa Zaire, brotes 2014-2016)
# ---------------------------------------------------------------------------
# R0 = 1.5-2.5. Contacto DIRECTO con fluidos. No aereo en condiciones nat.
# ell muy pequenio: simula radio de riesgo por salpicadura.
# lam extremo: CFR 25-90%.
VIRUS_CATALOG["ebola"] = VirusProfile(
    name          = "Ebola (Ebolavirus cepa Zaire)",
    description   = (
        "Fiebre hemorragica viral. Transmision exclusivamente por contacto "
        "directo con fluidos corporales. ell muy bajo simula radio de "
        "salpicadura. Alta mortalidad sin soporte intensivo."
    ),
    tau_max       = 28.0,    # requiere contacto fisico con alta carga viral
    ell           = 0.25,    # metros; casi contacto directo
    delta_ext     = 8.0,     # /dia exterior (T1/2 <2h fuera del huesped)
    delta_cerrado = 3.0,     # /dia cerrado (algo mas estable en superficies)
    delta_abierto = 15.0,    # /dia abierto (UV, temperatura lo inactivan rapido)
    lam           = 12.0,    # mortalidad extrema
    k_E           = 3,
    p_E           = 0.25,    # media E = 3*0.75/0.25 = 9.0 dias
    mu_R          = 730.0,   # ~2 anos de inmunidad robusta
    M_R           = 1095.0,  # cap 3 anos
    r0_ref        = "R0 = 1.5-2.5 (Althaus 2014; WHO 2014)",
    cfr_ref       = "CFR = 25-90% segun cepa y atencion (Feldmann & Geisbert 2011)",
    notes         = (
        "ell=0.25m: mecanismo de contacto modelado como aerosol de corto "
        "alcance (salpicadura). delta_ext=8.0: Sagripanti et al. 2010, "
        "T1/2 <1h en superficies/aire. lam=12 calibrado para CFR~50% "
        "con inmunidad adaptativa media."
    ),
)

# ---------------------------------------------------------------------------
# 6. TUBERCULOSIS (Mycobacterium tuberculosis cepa DS)
# ---------------------------------------------------------------------------
# R0 = 2-4 (condiciones modernas); 10-14 (hacinamiento).
# Aerosoles <5um de extrema persistencia. Dosis infecciosa ~1-10 bacilos.
VIRUS_CATALOG["tuberculosis"] = VirusProfile(
    name          = "Tuberculosis (M. tuberculosis DS)",
    description   = (
        "Enfermedad bacteriana cronica, segunda causa infecciosa de muerte "
        "global (WHO 2023). Transmision exclusivamente aerea por nucleos "
        "goticulares. Aqui se modela la progresion a TB pulmonar activa."
    ),
    tau_max       = 3.0,     # dosis infecciosa muy baja (~1-10 bacilos)
    ell           = 3.5,     # metros; aerosoles de larga persistencia
    delta_ext     = 0.08,    # /dia exterior; bacilo extremadamente resistente
    delta_cerrado = 0.05,    # /dia cerrado; persiste horas-dias sin ventilacion
    delta_abierto = 0.5,     # /dia abierto; algo mas susceptible a UV
    lam           = 4.5,     # mortalidad alta sin tratamiento
    k_E           = 4,
    p_E           = 0.12,    # media E = 4*0.88/0.12 = 29.3 dias (~1 mes)
    mu_R          = 1825.0,  # ~5 anos post-tratamiento
    M_R           = 3650.0,  # cap 10 anos
    r0_ref        = "R0 = 2-4 moderno; 10-14 en hacinamiento (Styblo 1991)",
    cfr_ref       = "CFR ~50% sin tratamiento (Murray 1990); <5% con tratamiento",
    notes         = (
        "k_E=4, p_E=0.12: incubacion media ~29 dias (latencia->TB activa). "
        "delta_cerrado=0.05: bacilo viable horas-dias en aire sin ventilacion "
        "(Escombe et al. 2009). tau_max=3.0: dosis infecciosa ~1-10 bacilos "
        "(Roth et al. 2004)."
    ),
)
# ---------------------------------------------------------------------------
# 7. VIH/SIDA (Human Immunodeficiency Virus)
# ---------------------------------------------------------------------------
# R0 = 2-5 (depende de la red de contacto). Infección crónica asintomática.
# Transmisión fluida (hemática/sexual). Ausencia de recuperación (R=0).
# Falla el SEIRSD convencional: requiere un modelo de cronicidad (I -> C).
VIRUS_CATALOG["hiv"] = VirusProfile(
    name            = "VIH/SIDA",
    description     = "Infección viral crónica con largo periodo de latencia. Afecta el sistema inmune debilitando la defensa contra infecciones oportunistas.",
    tau_max         = 50.0,    # Muy alta: fase crónica de años
    ell             = 0.0,     # Contacto directo/fluidos
    delta_ext       = 10.0,    # Inactivación rápida fuera del cuerpo
    delta_cerrado   = 10.0, 
    delta_abierto   = 10.0, 
    lam             = 0.0,     # Letalidad directa baja (muerte por complicaciones tardías)
    k_E             = 10,      # Periodo de incubación muy largo
    p_E             = 0.001,   # Transición a fase sintomática lenta
    mu_R            = 0.0,     # Sin recuperación: cronicidad (permanece en R o estado crónico)
    M_R             = 0.0,
    r0_ref          = "R0 ~ 2-5 (dependiendo de la red social/comportamiento)",
    cfr_ref         = "CFR ~ 80-90% sin tratamiento antirretroviral",
    notes           = "Modelo: Representar como S -> E -> I (crónico) donde la transición a R es casi nula."
)


# ---------------------------------------------------------------------------
# 8. SARS-CoV-1 (2003)
# ---------------------------------------------------------------------------
# R0 = 2-3. Transmisión hospitalaria dominante (nosocomial).
# Alta letalidad (CFR >9%) facilita la contención mediante aislamiento riguroso.
# El riesgo de superpropagación exige modelar "nodos" de alta densidad.
VIRUS_CATALOG["sars_2003"] = VirusProfile(
    name            = "SARS-CoV-1 (2003)",
    description     = "Coronavirus de alta patogenicidad, enfocado en entornos hospitalarios (eventos de superpropagación).",
    tau_max         = 10.0,
    ell             = 2.0,     # Transmisión en proximidad hospitalaria
    delta_ext       = 1.0,     # Estable en superficies
    delta_cerrado   = 0.5, 
    delta_abierto   = 2.0, 
    lam             = 9.6,     # CFR alta (9-10%)
    k_E             = 5,       # Incubación 2-7 días
    p_E             = 0.2, 
    mu_R            = 1000.0,  # Inmunidad robusta post-recuperación
    M_R             = 1000.0,
    r0_ref          = "R0 ~ 2.0-3.0",
    cfr_ref         = "CFR ~ 9.6% (WHO)",
    notes           = "Requiere alta restricción de contactos (aislamiento) para ser contenido."
)

# ---------------------------------------------------------------------------
# 9. FIEBRE HEMORRAGICA ARGENTINA (Virus Junin)
# ---------------------------------------------------------------------------
# R0 = 1.0 (aprox, endémico en zonas rurales). Transmision por contacto
# con excretas de roedores (Calomys musculinus). No aereo.
# CFR ~30% sin suero inmune; <1% con tratamiento (Maiztegui et al.).
VIRUS_CATALOG["junin"] = VirusProfile(
    name          = "Fiebre Hemorrágica Arg. (Virus Junin)",
    description   = (
        "Enfermedad viral endémica de la región pampeana. Transmision por "
        "inhalación de partículas infectadas en ambientes rurales. "
        "Calibrado con alta letalidad base y baja tasa de transmision persona-persona."
    ),
    tau_max       = 30.0,    # requiere exposicion ambiental directa
    ell           = 0.4,     # metros; contagio por proximidad a aerosoles de rastrojo
    delta_ext     = 2.0,     # /dia; relativamente estable en condiciones secas
    delta_cerrado = 1.0,     # /dia
    delta_abierto = 5.0,     # /dia
    lam           = 9.0,     # alta letalidad sin soporte serologico
    k_E           = 2,
    p_E           = 0.20,    # media E = 2*0.8/0.2 = 8.0 dias
    mu_R          = 3650.0,  # inmunidad prolongada (~10 años)
    M_R           = 3650.0,
    r0_ref        = "R0 ~ 1.0 (Endémico regional)",
    cfr_ref       = "CFR ~30% (sin tratamiento); <1% (con suero, Maiztegui 1979)",
    notes         = (
        "Modelado para entornos rurales (rastrojos). ell=0.4m: simula "
        "inhalacion de polvo contaminado en cercanía."
    ),
)

# Alias convenientes
VIRUS_CATALOG["sarampion"]   = VIRUS_CATALOG["measles"]
VIRUS_CATALOG["gripe"]       = VIRUS_CATALOG["influenza"]
VIRUS_CATALOG["gripe porcina"]       = VIRUS_CATALOG["h1n1_2009"]
VIRUS_CATALOG["covid"]       = VIRUS_CATALOG["covid19_delta"]
VIRUS_CATALOG["ebola_zaire"] = VIRUS_CATALOG["ebola"]
VIRUS_CATALOG["tb"]          = VIRUS_CATALOG["tuberculosis"]
VIRUS_CATALOG["vih/sida"]       = VIRUS_CATALOG["hiv"]
VIRUS_CATALOG["sars03"]       = VIRUS_CATALOG["sars_2003"]
VIRUS_CATALOG["fha"] = VIRUS_CATALOG["junin"]


# ===========================================================================
# Funciones de utilidad
# ===========================================================================

def apply_to_simulation(sim: Any, profile: VirusProfile) -> None:
    """
    Inyecta los parametros de un VirusProfile en una instancia de
    ContinuousSEIRSDSimulation (o cualquier subclase de BaseSEIRSDSimulation).

    Solo modifica atributos de configuracion biologica/fisica.
    NO toca el nucleo de la dinamica de Langevin ni los arrays de estado.
    Debe llamarse ANTES de sim.run() (idealmente antes de _fase0_inicializacion).

    Parameters
    ----------
    sim : ContinuousSEIRSDSimulation
        Instancia del simulador antes de llamar a run().
    profile : VirusProfile
        Perfil del VIRUS_CATALOG.

    Raises
    ------
    AttributeError
        Si el simulador no expone alguno de los atributos requeridos.

    Example
    -------
    >>> from virus_factory import VIRUS_CATALOG, apply_to_simulation
    >>> from continuous_simulation import ContinuousSEIRSDSimulation
    >>>
    >>> sim = ContinuousSEIRSDSimulation(N=1600, L=100.0, t_max=30)
    >>> apply_to_simulation(sim, VIRUS_CATALOG["measles"])
    >>> sim.run(n_seed=10, seed=42)
    """
    required = ("tau_max", "ell", "delta_ext", "delta_cerrado",
                 "delta_abierto", "lam", "k_E", "p_E", "mu_R", "M_R")

    missing = [a for a in required if not hasattr(sim, a)]
    if missing:
        raise AttributeError(
            f"El simulador no tiene los atributos: {missing}. "
            f"Revisa que sea una subclase de BaseSEIRSDSimulation."
        )

    sim.tau_max        = profile.tau_max
    sim.ell            = profile.ell
    sim.delta_ext      = profile.delta_ext
    sim.delta_cerrado  = profile.delta_cerrado
    sim.delta_abierto  = profile.delta_abierto
    sim.lam            = profile.lam
    sim.k_E            = profile.k_E
    sim.p_E            = profile.p_E
    sim.mu_R           = profile.mu_R
    sim.M_R            = profile.M_R


def list_diseases() -> list[str]:
    """Retorna las claves canonicas del catalogo (sin alias)."""
    seen: set[int] = set()
    canonical: list[str] = []
    for key, profile in VIRUS_CATALOG.items():
        if id(profile) not in seen:
            seen.add(id(profile))
            canonical.append(key)
    return canonical


def get_profile(key: str) -> VirusProfile:
    """
    Retorna un VirusProfile por clave. Levanta KeyError descriptivo si no existe.

    Example
    -------
    >>> p = get_profile("influenza")
    >>> print(p.mean_incubation_days)
    2.0
    """
    key_lower = key.lower().strip()
    if key_lower not in VIRUS_CATALOG:
        raise KeyError(
            f"Virus {key!r} no encontrado. "
            f"Disponibles: {list_diseases()}"
        )
    return VIRUS_CATALOG[key_lower]

def get_parameter_table():
    headers = ["Enfermedad", "tau_max", "ell (m)", "delta_ext", "lam", "Inc (d)", "mu_R (d)"]
    data = [
        ["Sarampion",    2.0,  4.0,  0.30, 1.5,  10.0,  36500],
        ["COVID Delta",  6.0,  2.5,  1.50, 3.5,  5.0,   120],
        ["Influenza",    15.0, 1.2,  4.00, 0.8,  2.0,   150],
        ["Ebola Zaire",  28.0, 0.25, 8.00, 12.0, 9.0,   730],
        ["Tuberculosis", 3.0,  3.5,  0.08, 4.5,  29.3,  1825],
        ["FHA (Junin)",  30.0, 0.4,  2.00, 9.0,  8.0,   3650]
    ]
    
    # "fancy_grid" es el estilo mas profesional y estetico
    return tabulate(data, headers=headers, tablefmt="fancy_grid", floatfmt=".2f")

# Para que no rompas tu código anterior, podés dejar esto:
PARAMETER_REFERENCE = get_parameter_table()