# Modelo Epidemiológico SEIRS-D
## Arquitectura Matemática Unificada
### Enfoques Discreto y Continuo con Núcleo Biológico Común (SDE-OU)

---

## Resumen de Arquitectura

Ambos enfoques comparten un núcleo biológico común regido por Ecuaciones Diferenciales Estocásticas (SDE-OU), pero difieren en su mecanismo de contagio.

| Componente | Enfoque Discreto (Grilla) | Enfoque Continuo (Browniano) |
|---|---|---|
| Mecanismo de contagio | Contacto directo agente-a-agente vía función de Hill σ(Vⱼ) con V₅₀ indexado por ωᵢₙ,ᵢ | Campo ambiental de inóculo Dⱼ(t) ≥ τᵢ |
| Rol de la carga viral Vⱼ | Entra en σ(Vⱼ) como probabilidad de transmisión directa | Alimenta la emisión de inóculo Iⱼ,ₜ al campo ambiental |
| Umbral de infección | Probabilidad por paso de tiempo — V₅₀(ωᵢₙ,ᵢ) heterogéneo por celda susceptible | τᵢ = ωᵢₙ,ᵢ · τ_max (derivado directamente de la cópula) |
| Dinámica viral interna | SDE-OU con atractor dinámico μ(τᵢ) y θ(τᵢ, ωₐd) — núcleo biológico común | SDE-OU con atractor dinámico μ(τᵢ) y θ(τᵢ, ωₐd) — núcleo biológico común |
| Cinemática de agentes | Celdas estáticas — grilla fija (cuadrada o hexagonal) | Partículas brownianas con difusividad acoplada a Vᵢ |
| Complejidad computacional | O(N · \|Nᵢ\|) con vecindad Moore o von Neumann | O(N log N) con árbol k-d para kernel de distancia |

---

## Parte I: Núcleo Biológico Universal

El núcleo biológico es compartido por ambos enfoques. Los agentes no son receptores pasivos sino sistemas biológicos con umbrales de respuesta individuales.

### 1. Vector de Vulnerabilidad Correlacionado

El estado inmunológico de cada agente se define por un vector bidimensional extraído de una Cópula Gaussiana que correlaciona la barrera innata con la capacidad adaptativa:

$$\mathbf{\Omega}_i = [\omega_{in}, \omega_{ad}]^T \in [0,1]^2 \qquad \text{con} \qquad \mathbf{\Omega}_i \sim C_\rho(\text{Beta}_{in}, \text{Beta}_{ad})$$

**Justificación:** La correlación entre ωᵢₙ (barrera innata) y ωₐd (capacidad adaptativa) refleja que el deterioro sistémico global es el factor determinante del riesgo individual. La independencia entre ambas variables sería una falacia epidemiológica.

### 2. Dinámica Viral — Proceso de Ornstein-Uhlenbeck con Atractor Dinámico

La dinámica viral interna sigue una SDE con reversión a la media. El tiempo de referencia es τᵢ = t − tᵢⁱⁿᶠ (tiempo transcurrido desde la infección del agente i), no el tiempo global t. Usar t global sincronizaría artificialmente los picos virales de toda la población infectada.

$$dv_i(t) = \theta(\tau_i, \omega_{ad}) \cdot \bigl(\mu(\tau_i, \omega_{ad}) - v_i(t)\bigr)\, dt + \sigma(\omega_{ad})\, dW_i^v(t)$$

#### 2.1 Atractor Dinámico

El atractor adopta una forma sigmoide indexada por τᵢ para capturar tanto la fase ascendente como la de resolución de la viremia:

$$\mu(\tau_i, \omega_{ad}) = v_{base} + \bigl(v_{peak}(\omega_{ad}) - v_{base}\bigr) \cdot \Phi(\tau_i)$$

$$\Phi(\tau_i) = 1 - e^{-k \cdot \tau_i}$$

**Comportamiento garantizado:**
- **Fase ascendente (τᵢ pequeño):** el atractor sube rápidamente arrastrando vᵢ hacia v_peak con estocasticidad controlada por σ.
- **Fase de resolución (τᵢ grande):** el sistema inmune reduce μ y θ alto fuerza la reversión a v_base.
- **Heterogeneidad poblacional:** v_peak(ωₐd) es función del perfil inmune, generando trayectorias virales individualizadas.

#### 2.2 θ Unificado — Fase × Capacidad Adaptativa

La tasa de reversión integra la estructura seccionada original con un factor multiplicativo de la capacidad adaptativa del individuo:

$$\theta(\tau_i, \omega_{ad}) = \theta_{fase}(\tau_i) \cdot (1 + \beta \cdot \omega_{ad})$$

donde la componente de fase mantiene la lógica biológica original:

$$\theta_{fase}(\tau_i) = \begin{cases} \theta_{low} & \text{si } \tau_i < \tau_{peak} \quad \text{[fase ascendente: alta variabilidad]} \\ \theta_{high} & \text{si } \tau_i \geq \tau_{peak} \quad \text{[resolución: convergencia forzada]} \end{cases}$$

**Impacto biológico del factor (1 + β·ωₐd):**
- **Fase ascendente (τᵢ < τ_peak):** θ_low es bajo, permitiendo la replicación viral hacia el pico. Un individuo con alta capacidad adaptativa (ωₐd → 1) ejerce un control estocástico ligeramente superior incluso en el inicio.
- **Fase de resolución (τᵢ ≥ τ_peak):** el sistema entra en θ_high amplificado. Un agente con ωₐd alto tendrá un θ masivo, forzando a vᵢ(t) a colapsar rápidamente hacia v_base. Un agente inmunodeprimido (ωₐd → 0) tendrá aclaramiento lento, manteniendo carga viral alta (y siendo supercontagiador) por más tiempo.

#### 2.3 Integración Exacta y Composición de Operadores (Step-Splitting)

Para evitar el error de discretización de Euler-Maruyama, se emplea la solución de transición gaussiana exacta del proceso OU:

$$v_{t+\Delta t} \sim \mathcal{N}\!\left( v_t \cdot e^{-\theta \Delta t} + M(\tau_i, \Delta t),\; \sigma^2 \frac{1 - e^{-2\theta \Delta t}}{2\theta} \right)$$

**Nota de implementación (Punto Medio y Splitting de Fase):**
1. $M(\tau_i, \Delta t)$ se evalúa en el punto medio del paso temporal para mayor precisión cuando $\mu(\tau_i)$ varía rápidamente en la fase ascendente.
2. **Mitigación de sesgo temporal (Step-Splitting):** Cuando un agente cruza el umbral de fase $\tau_{peak}$ durante el paso temporal (es decir, $0 < \tau_{peak} - \tau_i < \Delta t$), el paso se divide secuencialmente en $\Delta t_1 = \tau_{peak} - \tau_i$ (evolucionando con $\theta_{low}$) y $\Delta t_2 = \Delta t - \Delta t_1$ (evolucionando con $\theta_{high}$ desde el estado intermedio $v^*$).
3. **Cálculo de AUC Exacto:** Para los agentes con step-splitting, la dosis acumulada (AUC) del paso se calcula mediante la composición trapezoidal de dos etapas para evitar subestimar el pico máximo de carga viral:
   $$\Delta \text{AUC}_{\text{split}} = \frac{v_t + v^*}{2} \cdot \Delta t_1 + \frac{v^* + v_{next}}{2} \cdot \Delta t_2$$

---

## Parte II: Compartimentos SEIRS-D y Transiciones

Cada celda (discreto) o agente (continuo) se encuentra en uno de los siguientes estados:

- **S (Susceptible):** individuo sano propenso a contraer la infección.
- **E (Expuesto):** individuo incubando el patógeno, no infeccioso.
- **I (Infectado):** individuo con carga viral activa y capacidad de transmisión.
- **R (Recuperado):** individuo con inmunidad temporal.
- **D (Muerto):** estado terminal irreversible.

### 1. Tasa de Mortalidad Base (Ruido de Fondo)

Referencia demográfica: 8 decesos anuales por cada 1.000 habitantes (Banco Mundial). La simulación opera en pasos de tiempo diarios (Δt = 1 día):

$$P(\text{Muerte Anual}) = 0.008 \implies P(\text{Supervivencia Anual}) = 0.992$$

$$p_{base} = 1 - (0.992)^{1/365} \approx 2.19 \times 10^{-5}$$

Esta probabilidad diaria de transición a D por causas ajenas al virus se evalúa de forma independiente para cada celda/agente.

### 2. Período de Incubación: Transición E → I

El tiempo de permanencia en el estado E se modela mediante una Distribución Binomial Negativa con k = 2 etapas internas de replicación viral, evitando la falta de memoria biológica de la distribución geométrica:

$$X \sim \text{NegBin}(k = 2,\; p = 0.5)$$

$$P(X = t) = \binom{t+1}{t} (0.5)^2 (0.5)^t$$

### 3. Dinámica del Estado Infectado (I → Stay, I → R, I → D)

Al ingresar al estado I, la probabilidad de continuar enfermo disminuye exponencialmente en función del tiempo acumulado de infección τ:

$$P(\text{Stay}|\tau) = e^{-\alpha \tau} \qquad \implies \qquad P(\text{Salir}|\tau) = 1 - e^{-\alpha \tau}$$

**Umbral de letalidad individualizado.** Si la celda/agente sale del compartimento I en el tiempo τ_final, su destino se dirime mediante un umbral endogenizado que integra el daño acumulado por viremia, el desgaste temporal y la fragilidad adaptativa del individuo.

El **daño por viremia** se calcula como el Área Bajo la Curva de la trayectoria viral, normalizada al rango unitario:

$$\text{AUC}_i = \frac{1}{\text{AUC}_{max}} \int_0^{\tau_{final}} v_i(\tau)\, d\tau \in [0, 1]$$

El **desgaste temporal** se normaliza usando τ_max como escala de referencia:

$$\tilde{\tau}_{i} = \frac{\tau_{final}}{\tau_{max}} \in [0, 1]$$

El **Índice de Estrés Biológico Neto** combina ambas componentes mediante una combinación lineal convexa con pesos w₁ + w₂ = 1:

$$E_i = w_1 \cdot \text{AUC}_i + w_2 \cdot \tilde{\tau}_i$$

El **umbral de letalidad individualizado** μᵥ,ᵢ enfrenta el estrés neto Eᵢ ∈ [0, 1] contra la capacidad adaptativa ωₐd,ᵢ ∈ [0, 1], manteniéndose el argumento de la logística estrictamente en [−1, 1]:

$$\mu_{v,i} = \frac{1}{1 + e^{-\lambda(E_i - \omega_{ad,i})}} = \frac{1}{1 + e^{-\lambda\left(w_1 \cdot \text{AUC}_i + w_2 \cdot \frac{\tau_{final}}{\tau_{max}} - \omega_{ad,i}\right)}}$$

**Interpretación epidemiológica de los pesos:**

| Régimen | w₁ → 1, w₂ → 0 | w₁ → 0, w₂ → 1 |
|---|---|---|
| Tipo de patógeno | Citotóxico directo (Ébola, Dengue hemorrágico) | Desgaste / inmunopatológico (VIH avanzado, COVID-19 crónico) |
| Mecanismo letal | Destrucción tisular por carga viral acumulada | Fatiga orgánica y respuesta inflamatoria sostenida |
| Supervivencia al pico | Si se supera el pico, el tiempo adicional penaliza poco | Cada día de infección adicional incrementa el riesgo independientemente del nivel viral |

**Mecanismo de decisión al salir de I:**
1. Se evalúa U ~ Uniforme(0, 1).
2. Si U ≤ μᵥ,ᵢ → transición al estado **D** (Muerto).
3. Si U > μᵥ,ᵢ → transición al estado **R** (Recuperado).

**Consistencia del comportamiento:**
- *Paciente crónico vulnerable* (ωₐd → 0): nunca logra un pico masivo pero tampoco erradica el virus. AUCᵢ moderado, pero τ̃ᵢ crece monótonamente. El estrés Eᵢ sigue escalando aunque la carga viral sea baja, modelando la muerte por agotamiento sistémico.
- *Paciente fuerte con infección relámpago* (ωₐd → 1, τ_final pequeño): AUCᵢ bajo y τ̃ᵢ insignificante. Eᵢ ≈ 0, garantizando supervivencia consistente con la biología.

### 4. Pérdida de Inmunidad: Transición R → S

La degradación de anticuerpos se modela mediante una Binomial Negativa parametrizada por dos valores observables:

- μ: media global del período de inmunidad.
- M: moda (pico) de la distribución de pérdida de inmunidad.

Los parámetros k y p se obtienen resolviendo el sistema:

$$\mu = \frac{r(1-p)}{p}$$

$$M = \begin{cases} \left\lfloor \dfrac{(r-1)(1-p)}{p} \right\rfloor & \text{si } r > 1 \\ 0 & \text{si } r \le 1 \end{cases}$$

---

## Parte III: Mecanismo de Contagio — Enfoque Discreto

En el enfoque discreto, el contagio es exclusivamente directo: agente-a-agente. Cada celda susceptible evalúa la influencia de sus vecinas infectadas en la vecindad Nᵢᴵ (Moore o von Neumann). No existe campo ambiental ni acumulación de inóculo en este enfoque.

### 1. Función de Transmisibilidad — Hill con V₅₀ Heterogéneo

La probabilidad de transmisión desde una celda infectada j depende de su carga viral interna Vⱼ mediante una función de Hill. El umbral de saturación ya no es una constante global, sino que está indexado por la barrera innata de la celda **receptora** i:

$$\sigma(V_j, \omega_{in,i}) = \frac{V_j^n}{V_j^n + V_{50}(\omega_{in,i})^n}$$

$$V_{50}(\omega_{in,i}) = V_{50,basal} \cdot (1 + \gamma \cdot \omega_{in,i})$$

| Parámetro | Símbolo | Descripción |
|---|---|---|
| Carga viral basal al 50% | V₅₀,basal | Umbral de saturación para un individuo de barrera nula (ωᵢₙ = 0) |
| Escala de protección innata | γ | Amplificación del umbral por unidad de ωᵢₙ (γ > 0) |
| Exponente de Hill | n | Agudeza de la curva sigmoide (n > 2 = switch-like) |

**Interpretación:** Si ωᵢₙ,ᵢ → 1 (barrera fuerte), V₅₀ aumenta, exigiendo que los vecinos infectados tengan cargas virales mucho más altas para lograr el contagio. Si ωᵢₙ,ᵢ → 0, V₅₀ colapsa a su valor basal y el individuo es máximamente vulnerable.

**Justificación sobre la elección de Hill sobre sigmoide logística:** La sigmoide logística 1/(1 + e^(−Vⱼ)) nunca llega a cero — cualquier carga viral, por mínima que sea, genera probabilidad de transmisión positiva. Hill sí tiene umbral implícito: cuando Vⱼ ≪ V₅₀, σ ≈ 0, lo que refleja que cargas sub-umbral no transmiten.

**Nota:** V₅₀,basal, γ y n son parámetros exclusivos del enfoque discreto. En el discreto no existe acoplamiento cinemático, por lo que no hay función D_esp(Vᵢ) y estos parámetros no son compartidos.

### 2. Transición S → E

Asumiendo independencia estocástica entre los intentos de contagio de los vecinos, la probabilidad de que la celda i resulte expuesta en el tiempo t es:

$$P(S_i \to E_i) = 1 - \prod_{j \in \mathcal{N}_i^I} \bigl(1 - \sigma(V_j, \omega_{in,i})\bigr)$$

donde $\mathcal{N}_i^I$ es el conjunto de celdas vecinas a i activamente en estado I.

### 3. Topologías de Grilla

| Topología | Vecinos | Conectividad | Efecto esperado |
|---|---|---|---|
| Cuadrada — Von Neumann | 4 | Baja | Propagación más lenta; umbral epidémico más alto |
| Cuadrada — Moore | 8 | Media | Propagación diagonal habilitada |
| Hexagonal | 6 | Media-alta | Mayor área de contacto por celda; propagación isotrópica |

---

## Parte IV: Mecanismo de Contagio — Enfoque Continuo

En el enfoque continuo, el contagio ocurre exclusivamente a través del campo ambiental de inóculo. No existe acoplamiento directo entre agentes. La función de Hill σ(Vⱼ) no participa en la transmisión: Vⱼ alimenta directamente la emisión al campo ambiental.

### 1. Dinámica del Inóculo Ambiental

Cada región j acumula inóculo emitido por los agentes infectados en su vecindario N(j). Para conservar la masa ante el sesgo de convexidad de Jensen, la emisión emplea el Compensador de Itô:

$$I_{j,t} = \sum_{i \in N(j)} w(d_{ij}) \cdot \frac{\Delta t}{2} \cdot (V_{i,t} + V_{i,t+\Delta t}) \cdot \exp\!\left(\tfrac{1}{2}\sigma_i^2 \Delta t\right)$$

La acumulación de dosis con decaimiento ambiental analíticamente estable:

$$D_j(t + \Delta t) = D_j(t) \cdot e^{-\delta(\mathbf{x}_j, t)\, \Delta t} + I_{j,t}$$

**Decaimiento espacialmente variable — modelo de ventilación.** El parámetro de degradación $\delta$ no es un escalar global sino una función del tipo de espacio que ocupa la región $j$ en el tiempo $t$. Su fundamento físico es el modelo de Wells-Riley para transmisión aérea en recintos: en estado estacionario, la concentración de aerosoles infecciosos es inversamente proporcional al caudal de ventilación $Q$ del espacio. Un mayor caudal — viento, apertura, ventilación mecánica — acelera la eliminación del inóculo, equivalente a un $\delta$ alto.

$$\delta(\mathbf{x}_j, t) = \begin{cases} \delta_{cerrado} & \text{si } \mathbf{x}_j \in \text{hub cerrado activo en } t \\ \delta_{abierto} & \text{si } \mathbf{x}_j \in \text{hub abierto o zona gravitatoria activa en } t \\ \delta_{ext} & \text{exterior (default)} \end{cases}$$

con la jerarquía $\delta_{cerrado} < \delta_{ext} < \delta_{abierto}$, reflejando que:

- **Espacio cerrado** (escuela, oficina): recirculación de aire, sin exposición UV, ventilación limitada. El inóculo persiste — un infectado que ya abandonó el recinto puede seguir causando infecciones en quienes lleguen después (*transmisión diferida*).
- **Exterior** (default browniano): dispersión por convección y radiación UV. Decaimiento intermedio.
- **Espacio abierto** (plaza, zona gravitatoria): viento, dilución en volumen ilimitado, máxima exposición UV. El inóculo se degrada rápidamente — la transmisión requiere co-presencia casi simultánea.

La transmisión diferida en espacios cerrados — contagio entre personas que nunca estuvieron simultáneamente en el hub — es un fenómeno documentado para COVID-19, sarampión y tuberculosis que este modelo reproduce sin parametrización adicional: es el campo $D_j(t)$ con $\delta_{cerrado}$ bajo evaluado en las posiciones de los agentes que llegan *después* del infectado.

**Implementación.** En cada paso temporal, el valor de $\delta(\mathbf{x}_j, t)$ se determina verificando si la posición $\mathbf{x}_j$ cae dentro del radio de algún hub activo (`motion_state == 2` o zona gravitatoria con $|\mathbf{x}_j - \mathbf{c}_h| \leq r_h$) y aplicando el $\delta$ correspondiente. Las regiones fuera de todo hub usan $\delta_{ext}$.

**Nota sobre calibración:** Los tres valores deben calibrarse empíricamente o derivarse del caudal de ventilación del espacio. Valores de referencia para aerosoles respiratorios: $\delta_{cerrado} \in [0.1, 0.5]$ h⁻¹, $\delta_{ext} \in [0.5, 1.0]$ h⁻¹, $\delta_{abierto} \in [1.0, 4.0]$ h⁻¹.

### 2. Campo Ambiental en Espacio Continuo

Cada agente receptor i acumula la dosis absorbida mediante un kernel de distancia gaussiano:

$$D_i(t) = \int_0^t \left[\sum_{j \neq i} V_j(s) \cdot K\!\left(|\mathbf{x}_i(s) - \mathbf{x}_j(s)|\right)\right] ds$$

$$K(r) = \exp\!\left(-\frac{r^2}{2\ell^2}\right)$$

La evaluación eficiente usa un árbol k-d con radio de corte r_cut = 3ℓ, manteniendo complejidad O(N log N).

### 3. Umbral de Infección

El agente i se infecta cuando su dosis acumulada alcanza un umbral heterogéneo individual derivado directamente del valor de inmunidad innata extraído de la cópula (Parte I §1):

$$D_i(t) \geq \tau_i$$

$$\tau_i = \omega_{in,i} \cdot \tau_{max}$$

**Justificación:** Esta parametrización elimina la extracción redundante de la distribución Beta y utiliza el valor biológico correlacionado ya disponible. Un individuo sistémicamente deteriorado (ωᵢₙ → 0) tendrá un umbral de infección τᵢ mínimo; una barrera innata fuerte (ωᵢₙ → 1) requiere una dosis acumulada cercana a τ_max para infectarse. La heterogeneidad inter-individual queda completamente capturada por la distribución de ωᵢₙ en la cópula.

**Nota:** Este umbral es exclusivo del enfoque continuo. En el discreto, la protección de ωᵢₙ opera a través de V₅₀(ωᵢₙ,ᵢ), no mediante un umbral acumulado.

### 4. Cinemática — Langevin-Stratonovich

Los agentes son partículas brownianas cuya difusividad espacial se acopla con la carga viral interna mediante una función de Hill (inhibición motora):

$$d\mathbf{x}_i(t) = \sqrt{2\, D_{esp}(V_i(t))}\; \circ\; d\mathbf{W}_i^{esp}(t)$$

$$D_{esp}(V_i) = D_{basal}\left(1 - \frac{V_i^n}{V_i^n + V_{sint}^n}\right) + D_{min}$$

**Interpretación de Stratonovich (∘):** Es innegociable para ruido multiplicativo. La interpretación de Itô introduce un drift espurio proporcional a ∇D_esp, desplazando artificialmente los agentes hacia zonas de mayor difusividad. Stratonovich preserva la física correcta.

**Nota:** Los parámetros V_sint y n de D_esp son independientes de cualquier parámetro del enfoque discreto. Su rol es exclusivamente cinemático.

### 5. Integración — Operator Splitting con Predictor-Corrector

La rigidez numérica del sistema acoplado (viral + espacial) se resuelve con Operator Splitting:

1. **Operador Viral:** resolución analítica gaussiana exacta (SDE-OU, Parte I §2.3). Sin error de discretización.
2. **Predicción espacial:** paso completo con difusividad en tₙ → xᵢ*(tₙ₊₁).
3. **Evaluación del campo:** promedio entre posición actual y predicha → Īⱼ.
4. **Corrección:** dosis del agente receptor actualizada con Īⱼ, elevando precisión a O(Δt²).

---

## Parte V: Análisis de Datos y Métricas

### 1. Estimación del R₀

En la fase exponencial inicial (I(t) ≪ N), la tasa de crecimiento r se relaciona con R₀ según:

$$R_0 = 1 + r \cdot T_g$$

donde Tg es el tiempo generacional medio del patógeno (estimable del modelo OU como el tiempo al pico de Dⱼ). En el continuo, R₀ depende tanto de la persistencia viral (δ) como de la movilidad de los agentes (D_basal).

### 2. Análisis de Supervivencia — Kaplan-Meier

El tiempo hasta la infección Tᵢ = τᵢⁱⁿᶠ es el evento de interés. Las curvas de Kaplan-Meier estiman la función de supervivencia:

$$\hat{S}(t) = \prod_{t_j \leq t} \left(1 - \frac{d_j}{n_j}\right)$$

donde dⱼ es el número de nuevas infecciones en tⱼ y nⱼ el número de agentes aún susceptibles. Los agentes no infectados al final de la simulación son censuras derechas.

El análisis estratificado por perfil inmune (ωᵢₙ, ωₐd) permite identificar qué componente del vector de vulnerabilidad Ωᵢ domina el riesgo de infección temprana.

### 3. Dataset Multidimensional

Cada agente genera una trayectoria en panel indexado por (agente i, tiempo t):

- **Serie temporal espacial:** {xᵢ(t), yᵢ(t)} — solo en enfoque continuo.
- **Serie temporal biológica:** {vᵢ(t), Dᵢ(t), AUCᵢ(t)}.
- **Variables estáticas:** Ωᵢ, τᵢ_umbral (continuo), tᵢⁱⁿᶠ (o censura).

Este formato es compatible con modelos de Cox extendidos con covariables dependientes del tiempo.

---

## Parte VI: Restricciones y Límites Analíticos

### 1. Sensibilidad al Parámetro δ (Continuo)

El decaimiento ambiental δ es la principal fuente de incertidumbre del enfoque continuo. Su impacto es no lineal: valores bajos generan reservorios persistentes que sostienen la epidemia indefinidamente; valores altos producen transmisión casi instantánea convergiendo al límite de contacto directo. Se recomienda análisis de sensibilidad global (método de Sobol) antes de usar el modelo para inferencia.

### 2. Sesgo de Varianza — Compensador de Itô (Continuo)

La compensación por Itô en el inóculo ambiental sacrifica los momentos superiores de la distribución viral intra-paso para conservar el primer momento (masa viral total). En simulaciones donde la varianza intra-paso sea clínicamente relevante, se recomienda reducir Δt.

### 3. Consistencia Cinemática (Continuo)

La aparición de focos fantasma de infección persistentes es un indicador diagnóstico de Δt excesivamente grande para la velocidad de difusión. Reducir Δt o el radio de corte r_cut del kernel son las correcciones apropiadas.

### 4. Linaje Intrazable (Continuo)

Al tratar el inóculo como campo continuo aditivo, se pierde la capacidad de reconstruir árboles filogenéticos directos. Para estudios filogenéticos se requiere un modelo de contagio por contacto con registro de pares (emisor, receptor) — es decir, el enfoque discreto.

### 5. Topología vs. Dinámica (Discreto)

La elección de vecindad (Moore, Von Neumann, hexagonal) afecta el umbral epidémico y la velocidad de propagación de forma sistemática. Comparaciones entre topologías deben controlar el número efectivo de vecinos para aislar el efecto de la geometría de la conectividad.

### 6. Cálculo del AUCᵢ y Escala del Argumento Logístico (Ambos Enfoques)

El argumento de la función logística (AUCᵢ − ωₐd,ᵢ) mezcla magnitudes de distinta escala: AUCᵢ crece con la duración de la infección (potencialmente >> 1) mientras que ωₐd,ᵢ ∈ [0, 1]. Es imprescindible normalizar AUCᵢ al rango unitario — por ejemplo, dividiendo por AUC_max esperado bajo el patógeno simulado — antes de calibrar λ, o absorber la escala en λ de forma explícita durante la calibración.

---

## Parte VII: Protocolo de Inicialización y Setup de Agentes

En el tiempo t = 0, la población de tamaño N se inicializa de forma estructurada para garantizar la consistencia biológica y cinemática del espacio de estado. El algoritmo de setup ejecuta de manera secuencial los siguientes operadores para cada agente i ∈ {1, …, N}.

### 1. Mapeo del Perfil Inmune vía Algoritmo de Cópula

Para evitar la independencia artificial, el par Ωᵢ = [ωᵢₙ,ᵢ, ωₐd,ᵢ]ᵀ se genera mediante el método de inversión de la CDF:

**Paso 1.** Extracción del espacio latente gaussiano con correlación ρ:

$$\mathbf{Z}_i = [z_1, z_2]^T \sim \mathcal{N}\!\left(\mathbf{0},\, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)$$

**Paso 2.** Transformación a márgenes uniformes mediante la CDF estándar Φ:

$$u_{1,i} = \Phi(z_1) \qquad u_{2,i} = \Phi(z_2)$$

**Paso 3.** Proyección sobre las distribuciones biológicas objetivo:

$$\omega_{in,i} = F^{-1}_{\text{Beta}_{in}}(u_{1,i}) \qquad \omega_{ad,i} = F^{-1}_{\text{Beta}_{ad}}(u_{2,i})$$

### 2. Derivación de Parámetros de Frontera Deterministas

Una vez fijado Ωᵢ, las funciones de acoplamiento deterministas configuran el comportamiento de la entidad sin introducir ruido secundario descorrelacionado:

**Entorno Discreto:**
- Umbral de Hill receptor: V₅₀,ᵢ = V₅₀,basal · (1 + γ · ωᵢₙ,ᵢ)
- Posición estática fija asignada en la topología de grilla elegida.

**Entorno Continuo:**
- Umbral de tolerancia acumulada: τᵢ = ωᵢₙ,ᵢ · τ_max
- Coordenadas iniciales continuas: **x**ᵢ(0) ~ Uniforme([0, L]²)

### 3. Vector de Estado Inicial

Cada agente se registra con el siguiente vector de variables dinámicas en t = 0:

$$\mathbf{X}_i(0) = \bigl[\,\text{Estado}_i = S,\; t^{inf}_{i} = \emptyset,\; \tau_{interno,i} = 0,\; v_i(0) = 0,\; D_i(0) = 0,\; \text{AUC}_i(0) = 0\,\bigr]$$

### 4. Inyección del Patógeno (Siembra Epidémica)

Se define un subconjunto de pacientes cero I₀ ⊂ {1, …, N} con |I₀| ≪ N. Para todo agente m ∈ I₀, el vector de estado inicial se altera manualmente:

$$\text{Estado}_m = I, \qquad t^{inf}_{m} = 0, \qquad v_m(0) = v_{base} + \varepsilon$$

donde ε > 0 representa la siembra viral inicial. El resto de la población mantiene el vector de estado susceptible basal.

**Nota sobre la elección de I₀ y Criterio de Selección:** En la implementación de este modelo, el subconjunto de pacientes cero $I_0$ se selecciona mediante una **Siembra Aleatoria Uniforme (Random Uniform Seeding)** sin reemplazo sobre el total de la población de agentes ($N$). Esto se realiza fijando de manera determinista la semilla del generador pseudoaleatorio en `42` antes del muestreo para garantizar la reproducibilidad exacta de la simulación. Enfoques geográficamente concentrados (clústers) o de estratificación inmunológica no están activos por defecto en esta versión.

---

---

## Parte VIII: Extensiones Estructurales y Experimentos Canónicos

Esta sección formaliza dos ejes de extensión del modelo base: la introducción de **movilidad estructurada** (puntos de atracción y multiparche) y el **análisis de intervenciones de política sanitaria** (cuarentena con falla asintomática, distancia social). Cada extensión se define en términos del modelo existente, explicita qué parámetros modifica y propone el experimento canónico que la valida.

---

### VIII.A — Movilidad Estructurada

La movilidad de los agentes se organiza en torno a una jerarquía de cuatro tipos de espacio, que difieren no solo en frecuencia y duración de visita sino en la **composición del grupo de co-presentes** — la variable epidemiológicamente determinante:

| Tipo | Grupo co-presentes | $\lambda_{i,h}$ | $\kappa_h$ | Ejemplo |
|---|---|---|---|---|
| Residencial | Fijo (convivientes) | — | 0 | Hogar |
| Cerrado | Fijo (colegas, compañeros) | Alto | 0 | Trabajo, escuela |
| Abierto | Aleatorio (población general) | Moderado | 0 | Supermercado, transporte |
| Gravitatorio | Aleatorio (población general) | 0 | Alto | Plaza, centro histórico |

La distinción entre grupo fijo y aleatorio tiene consecuencias estructurales irreducibles a los parámetros de frecuencia y duración. Los hubs de grupo **cerrado** generan clusters saturables: una vez que el grupo está todo infectado o recuperado, el hub deja de ser vector activo. Los hubs de grupo **abierto** generan puentes entre clusters: conectan en cada visita subgrafos que de otro modo no tendrían contacto, acelerando la propagación a escala poblacional incluso cuando cada cluster individual está bajo control local. Esta es la diferencia entre $R_{ef}$ local (dentro del cluster) y $R_{ef}$ global (entre clusters).

La jerarquía tiene precedencia estricta: la agenda suspende la difusión; el hogar es el estado al que se regresa cuando no hay ningún evento activo.

---

#### VIII.A.0 Hogares y Transmisión Doméstica — Enfoque Continuo

**Motivación.** El modelo browniano base no distingue el tiempo que un agente pasa en su residencia del tiempo que pasa en cualquier otra zona del dominio. Sin embargo, la dinámica doméstica tiene propiedades epidemiológicas cualitativamente distintas: convivencia prolongada y nocturna, espacio cerrado, grupo pequeño con exposición mutua acumulada. Modelar el hogar como estado default — en lugar de como un hub más con $\lambda_{i,h}$ — captura esta asimetría: el agente *siempre* regresa al hogar, no lo visita.

**Asignación de hogares.** Al inicializar la simulación, los $N$ agentes se agrupan en $N_{hog}$ hogares. Cada hogar $k$ tiene una posición fija $\mathbf{h}_k$ en el dominio y un conjunto de convivientes $\mathcal{F}_k$ con $|\mathcal{F}_k| \sim \text{discreto}(\mathbf{p}_{tam})$, donde $\mathbf{p}_{tam}$ es la distribución de tamaño de hogar (calibrable desde datos censales). La asignación es permanente durante toda la simulación.

**Estado default.** El agente $i$ está en hogar cuando no tiene ningún evento de agenda activo ni tránsito en curso:

$$\mathbf{x}_i(t) = \mathbf{h}_{k(i)} \qquad \text{si } \nexists\, h : \texttt{tiempo\_restante\_visita}[i,h] > 0 \text{ y } \nexists \text{ tránsito activo}$$

Durante este estado el operador de Langevin se suspende, igual que en una visita de agenda. La diferencia es que no hay temporizador que expire — el agente permanece hasta que el siguiente evento de agenda lo saque.

**Agenda diaria.** El tiempo en hogar surge naturalmente de la diferencia entre el día completo y las salidas agendadas. Para modelar rutinas (sueño, comidas) sin necesidad de una agenda determinista compleja, se introduce un bloque de **permanencia nocturna** $[t_{noche}, t_{día}]$ durante el cual todos los eventos de agenda quedan bloqueados:

$$\lambda_{i,h}^{ef}(t) = \begin{cases} \lambda_{i,h} & t \in [t_{día},\, t_{noche}] \\ 0 & t \in [t_{noche},\, t_{día}] \end{cases}$$

con $t_{noche}$ y $t_{día}$ parámetros globales (por defecto 23:00 y 07:00). Esto garantiza que todos los agentes estén en hogar durante la noche sin necesidad de modelar el sueño explícitamente.

**Transmisión doméstica.** Los convivientes $\mathcal{F}_k$ comparten posición $\mathbf{h}_k$ durante las horas en hogar. El campo ambiental local $D_j(t)$ se evalúa en $\mathbf{h}_k$, acumulando el inóculo emitido por cualquier conviviente infectado. La transmisión doméstica no requiere ningún mecanismo adicional — emerge del campo continuo con la emisión escalada por:

$$\rho_{hogar} \geq 1$$

El factor $\rho_{hogar} > 1$ refleja que el espacio cerrado y la exposición prolongada producen concentraciones de inóculo mayores que en cualquier hub público. En términos relativos, $\rho_{hogar}$ normaliza la escala de $\rho_h$ de los otros hubs: fijar $\rho_{hogar} = 1$ y $\rho_h < 1$ para todos los hubs públicos es equivalente y más interpretable.

**Cuarentena doméstica y su paradoja.** Cuando un agente entra en estado de cuarentena ($Q_i = 1$, Parte VIII.B.1), se fija permanentemente en $\mathbf{h}_{k(i)}$ con todos sus eventos de agenda cancelados. Esto elimina la transmisión en hubs pero *mantiene* la exposición de los convivientes $\mathcal{F}_k$, que siguen compartiendo el campo ambiental del hogar. La cuarentena doméstica es efectiva para el exterior y contraproducente para el interior del hogar — un resultado que el modelo reproduce sin parametrización adicional y que es consistente con la evidencia observacional de COVID-19.

**Parámetros introducidos:**

| Parámetro | Símbolo | Rango típico |
|---|---|---|
| Número de hogares | $N_{hog}$ | $[N/5,\, N/2]$ |
| Distribución de tamaño de hogar | $\mathbf{p}_{tam}$ | Calibrar desde censo |
| Inicio del bloque nocturno | $t_{noche}$ | 22:00–24:00 |
| Fin del bloque nocturno | $t_{día}$ | 06:00–08:00 |
| Factor de emisión doméstica | $\rho_{hogar}$ | $[1.0, 3.0]$ |

---

#### VIII.A.1 Puntos de Atracción Central (Hubs) — Enfoque Continuo

El modelo de movimiento browniano isotrópico de la Parte IV §4 captura difusión residencial pero no la estructura de la movilidad humana real, que combina visitas sistemáticas con agenda propia y atracción gravitatoria pasiva hacia nodos densos. Estos mecanismos coexisten y requieren operadores separados. La taxonomía completa está en la introducción de VIII.A; aquí se formalizan los operadores.

**Parámetro de composición de grupo.** Cada hub $h$ tiene asignado un tipo de grupo:

$$\text{tipo}_h \in \{\text{cerrado},\, \text{abierto}\}$$

Para hubs cerrados, cada agente $i$ pertenece a un grupo fijo $G_{i,h}$ (análogo a `hogar_id` pero para el hub). El campo ambiental durante la visita se evalúa únicamente entre miembros de $G_{i,h}$ co-presentes en $\mathbf{c}_h$. Para hubs abiertos, el campo se evalúa entre todos los agentes presentes en $\mathbf{c}_h$ en ese instante, independientemente de su identidad — el conjunto de co-presentes varía en cada visita.

En implementación:

```
hub_group_id[i, h]  # entero fijo si hub cerrado; None si hub abierto
```

**Cuarentena en hub cerrado.** Cuando un agente de grupo cerrado entra en $Q_i = 1$, su ausencia permanente del hub reduce el tamaño efectivo del grupo expuesto. A diferencia de la cuarentena doméstica (que mantiene la transmisión intrahogar), la cuarentena en hub cerrado es neta: el grupo $G_{i,h} \setminus \{i\}$ queda con un miembro menos en exposición mutua. Si el agente en cuarentena era el único infectado del grupo, el hub cerrado queda efectivamente saneado.

**Parámetros de hub:**

- $\lambda_{i,h} \geq 0$: tasa de visita agendada (visitas por unidad de tiempo). Cero si el agente no tiene agenda en ese hub.
- $\kappa_h \geq 0$: masa gravitatoria. Cero para hubs de agenda pura.
- $\text{tipo}_h$: composición de grupo (cerrado / abierto).
- $G_{i,h}$: identificador de grupo fijo (solo hubs cerrados).

---

**Operador 1 — Agenda de visitas (Poisson).** Los tiempos entre visitas sucesivas al hub $h$ para el agente $i$ siguen:

$$T_{i,h}^{(k)} \sim \text{Exp}(\lambda_{i,h})$$

La duración de cada estadía se extrae de:

$$\Delta_{i,h} \sim \text{Gamma}(\alpha_h, \beta_h)$$

con $\alpha_h, \beta_h$ calibrados por tipo de hub ($\alpha_h \approx 1$ para visitas cortas variables como un kiosco; $\alpha_h \gg 1$ para jornadas de duración concentrada como una escuela).

El comportamiento durante la estadía depende del tipo de hub:

**Hubs cerrados — posición fija.** El operador de Langevin se suspende completamente y la posición se ancla en el centroide:

$$\mathbf{x}_i(t) = \mathbf{c}_h \qquad t \in \bigl[T_{i,h}^{(k)},\; T_{i,h}^{(k)} + \Delta_{i,h}\bigr]$$

Apropiado para escuelas y oficinas, donde el agente ocupa un puesto fijo y la exposición entre compañeros es uniforme y sostenida.

**Hubs abiertos — difusión local restringida.** El operador de Langevin se reactiva pero confinado al disco de radio $r_h$ centrado en $\mathbf{c}_h$:

$$d\mathbf{x}_i(t) = \sqrt{2\,D_{loc,h}}\;\circ\;d\mathbf{W}_i(t), \qquad |\mathbf{x}_i(t) - \mathbf{c}_h| \leq r_h$$

con $D_{loc,h} \leq D_{basal}$ una difusividad local reducida (el agente pasea, no difunde libremente) y la restricción de frontera implementada por reflexión elástica. Esta variante captura que en una plaza o centro comercial los agentes se mueven dentro del espacio sin abandonarlo durante la estadía, distribuyendo el inóculo dentro del radio en lugar de concentrarlo en un punto.

El campo ambiental $D_j(t)$ usa en ambos casos la posición actual del agente $\mathbf{x}_i(t)$ — en el caso de posición fija coincide con $\mathbf{c}_h$; en el caso de difusión restringida varía dentro del disco. El parámetro `motion_state` toma valor `2` en ambos sub-estados; la distinción se implementa mediante el flag `hub_difusion_local[h]`.

Al finalizar la estadía, el agente retoma la SDE global desde su posición actual $\mathbf{x}_i(T_{i,h}^{(k)} + \Delta_{i,h})$. El cierre de cualquier hub de agenda se implementa anulando la tasa: $\lambda_{i,h} \to 0\;\forall i$.

---

**Operador 2 — Atracción gravitatoria (deriva continua).** Fuera de cualquier visita agendada activa, la SDE espacial incorpora una deriva determinista hacia los hubs gravitatorios:

$$d\mathbf{x}_i(t) = \underbrace{\sum_{h=1}^H \kappa_h \cdot g\!\left(\mathbf{x}_i(t), \mathbf{c}_h\right) dt}_{\text{deriva gravitatoria}} + \underbrace{\sqrt{2\,D_{esp}(V_i(t))}\;\circ\;d\mathbf{W}_i^{esp}(t)}_{\text{difusión browniana}}$$

El kernel gravitatorio es gaussiano suavizado para evitar singularidades en $\mathbf{x}_i \approx \mathbf{c}_h$:

$$g(\mathbf{x}_i, \mathbf{c}_h) = \frac{\mathbf{c}_h - \mathbf{x}_i}{|\mathbf{c}_h - \mathbf{x}_i| + \epsilon} \cdot \exp\!\left(-\frac{|\mathbf{c}_h - \mathbf{x}_i|^2}{2\ell_h^2}\right)$$

donde $\ell_h$ es el radio de influencia del hub (más allá de $3\ell_h$ la atracción es despreciable) y $\epsilon$ es un regularizador numérico pequeño. Un hub gravitatorio no fija la posición del agente: el agente es atraído pero sigue difundiendo, modelando el comportamiento de quien deambula por una zona céntrica sin destino fijo.

**Precedencia de operadores.** Cuando el agente $i$ está en visita agendada activa a algún hub, el operador gravitatorio se inhibe — la posición está fijada y no hay SDE activa. La agenda tiene precedencia sobre la gravedad.

---

**Mecanismo de supercontagio.** En ambos operadores, la emisión de inóculo opera normalmente durante la presencia en el hub. La concentración de agentes en $\mathbf{c}_h$ — ya sea por agenda simultánea o por atracción gravitatoria — eleva transitoriamente el campo ambiental local $D_j(t)$ (Parte IV §1–2) muy por encima del nivel de fondo. Un infectado en hub de agenda emite en posición fija durante $\Delta_{i,h}$, maximizando la dosis acumulada por los co-presentes. Un infectado gravitatorio emite mientras deambula en el entorno del hub, con efecto algo más difuso pero sostenido en el tiempo. Ambos producen eventos de supercontagio que el browniano isotrópico no puede reproducir.

**Ajuste de emisión por tipo de hub.** Para reflejar ventilación, densidad y tiempo de exposición propios de cada espacio, la emisión del agente $i$ en hub $h$ se escala:

$$V_i^{hub}(t) = \rho_h \cdot V_i(t) \qquad \rho_h \in (0, 1]$$

con $\rho_h$ calibrable por arquetipo (escuela cerrada $\rho_h \approx 0.8$, plaza abierta $\rho_h \approx 0.3$).

---

**Experimento canónico 1 — Higiene vs. cierre de hubs.** Se comparan dos intervenciones activadas en $t = t^*$ (umbral de casos):

| Intervención | Parámetro modificado | Interpretación |
|---|---|---|
| Higiene poblacional | $V_{50,basal} \to \alpha_{hig} \cdot V_{50,basal},\; \alpha_{hig} > 1$ | Barrera innata global elevada |
| Cierre de hubs de agenda | $\lambda_{i,h} \to 0\;\forall i, h$ | Eliminación de visitas sistemáticas |
| Cierre de hubs gravitatorios | $\kappa_h \to 0\;\forall h$ | Eliminación de zonas de atracción pasiva |

La hipótesis es que el cierre de hubs de agenda reduce $R_{ef}$ de forma más abrupta que la higiene, al eliminar los eventos de supercontagio de alta densidad y duración fija. El cierre gravitatorio tiene efecto más gradual — los agentes siguen difundiendo en la zona pero sin sesgo — y puede ser menos efectivo si la concentración residual sigue siendo alta. La comparación tiene una asimetría temporal relevante: la higiene es activable instantáneamente a nivel individual, mientras que el cierre de hubs requiere decisión institucional con demora $\delta t_{inst}$ modelable como parámetro adicional.

---

#### VIII.A.2 Modelo Multiparche con Probabilidad de Viaje

El modelo base opera en una única región espacial de tamaño $L \times L$. Para representar dinámicas interpoblacionales (ciudades, barrios, países) se extiende a $P$ parches $\{p\}_{p=1}^P$, cada uno con su propia grilla o espacio continuo y población $N_p$.

**Operador de migración.** En cada paso de tiempo, cada agente $i$ en parche $p$ puede migrar al parche $q \neq p$ con probabilidad:

$$m_{pq}(t) = m_{pq}^0 \cdot \Pi(t)$$

donde $m_{pq}^0$ es la tasa de movilidad basal (estimable de matrices de origen-destino de transporte) y $\Pi(t) \in [0,1]$ es un factor de reducción de movilidad global activable en respuesta a la situación epidémica:

$$\Pi(t) = \begin{cases} 1 & \text{si } I_{total}(t) < I^* \\ \pi_{min} & \text{si } I_{total}(t) \geq I^* \end{cases}$$

con $I^*$ el umbral de casos que activa la restricción de movimiento y $\pi_{min} \in [0,1]$ la movilidad residual permitida (nunca cero: hay movilidad esencial).

Al migrar, el agente conserva su vector de estado completo $\mathbf{X}_i$ (compartimento, carga viral, AUC, Ωᵢ). El parche destino hereda el agente con su biología intacta.

**Mecanismo de acoplamiento.** Sin migración ($m_{pq}^0 = 0$), los parches evolucionan como epidemias independientes. La migración introduce acoplamiento: un parche con $I_p \ll 1$ puede recibir un agente infectado y encender una epidemia secundaria. La dinámica de parche $p$ en el enfoque continuo extiende el campo ambiental $D_j(t)$ al dominio espacial local de ese parche — los migrantes emiten inóculo en el nuevo espacio desde el momento de llegada.

**Experimento canónico 2 — Supervivencia diferencial de parches.** Se simulan $P = 5$ parches heterogéneos en densidad poblacional $N_p$, con un único parche semilla $p^*$ infectado en $t = 0$. Se registran:

- El tiempo de llegada de la epidemia a cada parche $T_{arr,p}$ (primera infección local).
- La fracción de parches que escapan la ola principal $\{p : I_p^{peak} < \varepsilon\}$ para $\varepsilon$ pequeño.
- El efecto de activar $\Pi(t)$ en $t^*$ sobre ambas métricas.

La hipótesis es que parches periféricos con $m_{p^*,p}^0$ bajo y activación temprana de $\Pi(t)$ tienen probabilidad no despreciable de escapar la epidemia — un resultado sensible al timing: la misma restricción de movilidad aplicada $\Delta t$ más tarde puede ser ineficaz si el parche ya recibió el caso índice.

---

### VIII.B — Intervenciones de Política Sanitaria

#### VIII.B.1 Cuarentena con Falla por Asintomaticidad

El compartimento E (expuesto) del modelo actual es epidemiológicamente silente: el agente es indetectable. La intervención de aislamiento actúa sobre agentes en estado I, pero la fracción asintomática de infectados también permanece indetectable durante parte de su curso clínico. Esta sección formaliza el mecanismo de cuarentena con falla estocástica.

**Definición de asintomaticidad.** Un agente $i$ en estado I es asintomático en el tiempo $t$ si su carga viral está por debajo de un umbral perceptible:

$$A_i(t) = \mathbb{1}[v_i(t) < v_{sint}]$$

donde $v_{sint}$ es el mismo umbral de inhibición motora de la Parte IV §4. Esta elección es deliberada: un agente cuya carga viral no alcanza para inhibir su movimiento tampoco presenta síntomas detectables, unificando los umbrales de detección clínica y efecto cinemático.

**Operador de cuarentena.** En $t \geq t^*$, se evalúa en cada paso para cada agente en estado I:

$$Q_i(t) = \begin{cases} 1 \quad \text{(cuarentena)} & \text{con probabilidad } (1 - A_i(t)) \cdot p_Q \\ 0 \quad \text{(libre)} & \text{en caso contrario} \end{cases}$$

donde $p_Q \in [0,1]$ es la eficacia del sistema de detección y aislamiento para casos sintomáticos. Los agentes con $Q_i = 1$ son excluidos de la vecindad de Hill (discreto) o fijados en posición aislada $\mathbf{x}^Q$ con emisión de inóculo bloqueada (continuo).

La falla del sistema tiene dos fuentes independientes:

1. **Falla por asintomaticidad:** $A_i(t) = 1$ — el agente no puede ser detectado.
2. **Falla del sistema:** $1 - p_Q$ — el sistema no captura al caso sintomático (capacidad limitada de trazado, demora en resultados, rechazo voluntario).

**Experimento canónico 3 — Velocidad de erradicación vs. fracción asintomática.** Se barre $v_{sint} \in [0.1, 0.9]$ (variando la definición de asintomaticidad) manteniendo $p_Q = 1$ para aislar el efecto. El resultado esperado es una curva de $T_{erad}$ (tiempo hasta $I_{total} = 0$) no monotónica en $v_{sint}$: un umbral de detección demasiado alto clasifica a muchos infectados como asintomáticos y falla; demasiado bajo genera cuarentenas excesivas con costo logístico. El punto de operación óptimo minimiza la integral $\int_0^{T_{erad}} I(t)\,dt$ sujeto a $\int_0^{T_{erad}} Q(t)\,dt \leq Q_{max}$ (capacidad de cuarentena).

---

#### VIII.B.2 Distancia Social como Parámetro de Intervención

La distancia social no es un compartimento sino una modificación de los parámetros de interacción existentes. Su efecto opera en mecanismos distintos según el enfoque:

**Enfoque discreto.** La distancia social eleva el umbral de Hill receptor de la totalidad de la población no-infectada:

$$V_{50}^{DS}(\omega_{in,i}) = V_{50}(\omega_{in,i}) \cdot (1 + \eta \cdot c_{DS})$$

donde $c_{DS} \in [0,1]$ es el nivel de cumplimiento de la medida (1 = cumplimiento total) y $\eta > 0$ es la eficacia biológica de la intervención. Esta formulación es consistente con la parametrización de higiene del experimento canónico 1, siendo la diferencia que $\eta \cdot c_{DS}$ puede estimarse de datos observacionales de contacto (encuestas, Bluetooth proximity, etc.).

**Enfoque continuo.** La distancia social reduce la difusividad basal de agentes sanos, contrayendo el radio de encuentro:

$$D_{basal}^{DS} = D_{basal} \cdot (1 - c_{DS} \cdot \eta_{mov})$$

con $\eta_{mov} \in (0, 1)$ para garantizar $D_{basal}^{DS} > 0$. La reducción de difusividad comprime la distribución de distancias inter-agente, reduciendo la integral del kernel gaussiano $K(r)$ en el campo de dosis.

**Cumplimiento parcial y heterogeneidad.** El caso de interés no es $c_{DS} = 1$ (trivial: la epidemia colapsa rápidamente) sino $c_{DS} < 1$ con fracción $1 - c_{DS}$ de la población no cumplidora. Para implementarlo correctamente, $c_{DS}$ se asigna a nivel individual:

$$c_{DS,i} \sim \text{Bernoulli}(C_{DS}) \qquad C_{DS} \in [0,1]$$

donde $C_{DS}$ es el nivel de cumplimiento poblacional. Los agentes con $c_{DS,i} = 0$ no modifican su comportamiento.

**Experimento canónico 4 — Umbral de cumplimiento y efecto de los hubs.** Se barre $C_{DS} \in [0, 1]$ en pasos de $0.05$ registrando el pico de prevalencia $I^{peak}$ y el total de muertos $D_{final}$. Se ejecuta la simulación en dos configuraciones: **sin hubs** y **con hubs activos**.

La pregunta experimental es: **¿los hubs anulan el efecto de la distancia social?** La hipótesis es que los hubs actúan como cortocircuito del mecanismo de distancia social: agentes que se repelen en espacio abierto convergen forzosamente en el hub, restaurando las condiciones de alta densidad de contacto. El experimento cuantifica el umbral de $C_{DS}$ necesario para que la distancia social sea efectiva *en presencia* de hubs, esperando que este umbral sea significativamente mayor que en su ausencia.

---

### VIII.C — Número Reproductivo Efectivo como Métrica de Intervención

El R₀ de la Parte V §1 es una estimación de fase exponencial. Para monitorear el efecto de las intervenciones en tiempo real se requiere el **número reproductivo efectivo** $R_{ef}(t)$, que es el promedio de contagios secundarios generados por un infectado en el estado actual de la población.

**Estimador de generación.** Para cada agente $i$ en estado I al tiempo $t$, se registra:

- $c_i^{hist}(t)$: número de agentes que $i$ ha infectado confirmadamente hasta $t$.
- $\hat{c}_i^{fut}(t)$: proyección de contagios futuros estimada por la fracción de vida infecciosa restante:

$$\hat{c}_i^{fut}(t) = c_i^{hist}(t) \cdot \frac{P(\tau_i > t)}{P(\tau_i \leq t)}$$

donde la distribución de $\tau_i$ se estima del proceso OU calibrado. El estimador poblacional es:

$$\hat{R}_{ef}(t) = \frac{1}{|I(t)|} \sum_{i \in I(t)} \left(c_i^{hist}(t) + \hat{c}_i^{fut}(t)\right)$$

**Clasificación dinámica de la epidemia:**

| Condición | Régimen |
|---|---|
| $\hat{R}_{ef}(t) > 1$ | Epidémico — la incidencia crece |
| $\hat{R}_{ef}(t) = 1$ | Endémico — incidencia estacionaria |
| $\hat{R}_{ef}(t) < 1$ | Declive — la epidemia se extingue |

$\hat{R}_{ef}(t)$ es la métrica canónica para evaluar si una intervención (cuarentena, distancia social, cierre de hubs) ha cruzado el umbral de control. Toda intervención de las secciones VIII.A y VIII.B debe reportarse con la trayectoria temporal de $\hat{R}_{ef}(t)$ antes y después de $t^*$ para permitir comparaciones de velocidad de respuesta entre estrategias.

---

## Apéndice: Tabla de Parámetros

### Parámetros del Núcleo Biológico (Ambos Enfoques)

| Parámetro | Símbolo | Rango típico | Descripción |
|---|---|---|---|
| Velocidad de reversión OU (ascenso) | θ_low | [0.1, 1] | Componente de fase base en τᵢ < τ_peak |
| Velocidad de reversión OU (resolución) | θ_high | [2, 10] | Componente de fase base en τᵢ ≥ τ_peak |
| Amplificación adaptativa de θ | β | [0.5, 5] | Escala del factor (1 + β·ωₐd); β = 0 recupera el modelo sin inmunidad adaptativa |
| Volatilidad viral | σ(ωₐd) | [0.05, 0.5] | Función del perfil inmune adaptativo |
| Velocidad de saturación del atractor | k | [0.5, 3.0] h⁻¹ | Velocidad de subida de μ(τᵢ) |
| Tasa de remisión viral | α | Variable por virus | Parámetro cinético de remisión en estado I |
| Sensibilidad del umbral de letalidad | λ | [1, 10] | Pendiente de la función logística μᵥ,ᵢ |
| Peso del daño viral (estrés) | w₁ | [0, 1] | Contribución de AUCᵢ al índice Eᵢ; w₁ + w₂ = 1 |
| Peso del desgaste temporal (estrés) | w₂ | [0, 1] | Contribución de τ̃ᵢ al índice Eᵢ; w₁ + w₂ = 1 |

### Parámetros Exclusivos del Enfoque Discreto

| Parámetro | Símbolo | Rango típico | Descripción |
|---|---|---|---|
| Carga viral basal al 50% de transmisión | V₅₀,basal | [0.3, 0.7] | Umbral de σ para un receptor con ωᵢₙ = 0 |
| Escala de protección innata (discreto) | γ | [0.5, 3] | Amplificación de V₅₀ por unidad de ωᵢₙ,ᵢ |
| Exponente de Hill (transmisión) | n | [1, 4] | Agudeza de la curva sigmoide de transmisión |

### Parámetros Exclusivos del Enfoque Continuo

| Parámetro | Símbolo | Rango típico | Descripción |
|---|---|---|---|
| Decaimiento ambiental en espacio cerrado | $\delta_{cerrado}$ | [0.1, 0.5] h⁻¹ | Tasa de degradación del inóculo en hubs cerrados (escuela, oficina) |
| Decaimiento ambiental en exterior | $\delta_{ext}$ | [0.5, 1.0] h⁻¹ | Tasa de degradación en zona browniana default |
| Decaimiento ambiental en espacio abierto | $\delta_{abierto}$ | [1.0, 4.0] h⁻¹ | Tasa de degradación en hubs abiertos y zonas gravitatorias |
| Longitud de escala del kernel | ℓ | [0.5, 5.0] m | Radio de dispersión del inóculo |
| Radio de corte del árbol k-d | r_cut | 3ℓ | Umbral de evaluación del kernel gaussiano |
| Difusividad basal de agentes | D_basal | [0.1, 10] m²/h | Movilidad de agentes sanos |
| Difusividad mínima | D_min | [0.01, 0.1] m²/h | Movilidad residual de agentes enfermos |
| Umbral de inhibición motora | V_sint | [0.3, 0.7] | Carga viral al 50% de inhibición de D_esp |
| Exponente de Hill (inhibición motora) | n_mov | [1, 4] | Agudeza de inhibición cinemática |
| Dosis máxima de infección | τ_max | Variable por virus | Dosis máxima acumulada; τᵢ = ωᵢₙ,ᵢ · τ_max |

### Tabla de Parámetros — Extensiones (Parte VIII)

| Parámetro | Símbolo | Rango típico | Sección |
|---|---|---|---|
| Número de hogares | $N_{hog}$ | $[N/5, N/2]$ | VIII.A.0 |
| Distribución de tamaño de hogar | $\mathbf{p}_{tam}$ | Desde censo | VIII.A.0 |
| Inicio del bloque nocturno | $t_{noche}$ | 22:00–24:00 | VIII.A.0 |
| Fin del bloque nocturno | $t_{día}$ | 06:00–08:00 | VIII.A.0 |
| Factor de emisión doméstica | $\rho_{hogar}$ | $[1.0, 3.0]$ | VIII.A.0 |
| Número de hubs | H | [1, 20] | VIII.A.1 |
| Tipo de grupo del hub $h$ | $\text{tipo}_h$ | cerrado / abierto | VIII.A.1 |
| Identificador de grupo fijo del agente $i$ en hub $h$ | $G_{i,h}$ | entero (hubs cerrados) | VIII.A.1 |
| Tasa de visita del agente $i$ al hub $h$ | $\lambda_{i,h}$ | [0, 5] visitas/día | VIII.A.1 |
| Forma de la distribución de estadía en hub $h$ | $\alpha_h$ | [1, 10] | VIII.A.1 |
| Escala de la distribución de estadía en hub $h$ | $\beta_h$ | [0.1, 2.0] h | VIII.A.1 |
| Radio de difusión local en hub abierto $h$ | $r_h$ | [0.05L, 0.2L] | VIII.A.1 |
| Difusividad local dentro del hub abierto $h$ | $D_{loc,h}$ | $[0, D_{basal}]$ | VIII.A.1 |
| Flag de difusión local del hub $h$ | $\texttt{hub\_difusion\_local}_h$ | booleano | VIII.A.1 |
| Masa gravitatoria del hub $h$ | $\kappa_h$ | [0, 5.0] | VIII.A.1 |
| Radio de influencia gravitatoria del hub $h$ | $\ell_h$ | [0.05L, 0.3L] | VIII.A.1 |
| Factor de emisión de inóculo en hub $h$ | $\rho_h$ | [0.1, 1.0] | VIII.A.1 |
| Factor de amplificación de higiene | $\alpha_{hig}$ | [1.5, 5.0] | VIII.A.1 |
| Número de parches | P | [2, 20] | VIII.A.2 |
| Tasa de migración basal entre parches | $m_{pq}^0$ | [0, 0.05] diaria | VIII.A.2 |
| Umbral de casos para restricción de viaje | $I^*$ | Variable | VIII.A.2 |
| Movilidad residual bajo restricción | $\pi_{min}$ | [0.05, 0.3] | VIII.A.2 |
| Umbral de detección sintomática | $v_{sint}$ | [0.1, 0.9] | VIII.B.1 |
| Eficacia del sistema de cuarentena | $p_Q$ | [0, 1] | VIII.B.1 |
| Capacidad máxima de cuarentena | $Q_{max}$ | Variable | VIII.B.1 |
| Eficacia biológica de distancia social (discreto) | $\eta$ | [0.5, 3.0] | VIII.B.2 |
| Eficacia cinemática de distancia social (continuo) | $\eta_{mov}$ | [0.1, 0.9] | VIII.B.2 |
| Nivel de cumplimiento poblacional | $C_{DS}$ | [0, 1] | VIII.B.2 |