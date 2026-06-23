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

#### 2.3 Integración Exacta — Transición Gaussiana

Para evitar el error de discretización de Euler-Maruyama, se emplea la solución de transición gaussiana exacta del proceso OU:

$$v_{t+\Delta t} \sim \mathcal{N}\!\left( v_t \cdot e^{-\theta \Delta t} + M(\tau_i, \Delta t),\; \sigma^2 \frac{1 - e^{-2\theta \Delta t}}{2\theta} \right)$$

**Nota de implementación:** M(τᵢ, Δt) debe evaluarse en el punto medio del paso temporal para mayor precisión cuando μ(τᵢ) varía rápidamente en la fase ascendente. El valor de θ usado aquí es θ(τᵢ, ωₐd) evaluado al inicio del paso.

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

$$D_j(t + \Delta t) = D_j(t) \cdot e^{-\delta \Delta t} + I_{j,t}$$

**Nota sobre δ:** El parámetro de degradación ambiental debe calibrarse empíricamente. Valores típicos para aerosoles en espacios cerrados: δ ∈ [0.1, 2.0] h⁻¹. Un δ mal estimado puede hacer al modelo irrealmente infeccioso (δ → 0) o instantáneamente no infeccioso (δ → ∞).

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

**Nota sobre la elección de I₀:** La selección del subconjunto semilla no es trivial. Una siembra aleatoria uniforme produce dinámicas iniciales distintas a una siembra geográficamente concentrada (clúster) o estratificada por perfil inmune. Para reproducibilidad, se recomienda fijar la semilla del generador pseudoaleatorio y documentar explícitamente el criterio de selección de I₀.

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
| Decaimiento ambiental | δ | [0.1, 2.0] h⁻¹ | Tasa de degradación del virus en el entorno |
| Longitud de escala del kernel | ℓ | [0.5, 5.0] m | Radio de dispersión del inóculo |
| Radio de corte del árbol k-d | r_cut | 3ℓ | Umbral de evaluación del kernel gaussiano |
| Difusividad basal de agentes | D_basal | [0.1, 10] m²/h | Movilidad de agentes sanos |
| Difusividad mínima | D_min | [0.01, 0.1] m²/h | Movilidad residual de agentes enfermos |
| Umbral de inhibición motora | V_sint | [0.3, 0.7] | Carga viral al 50% de inhibición de D_esp |
| Exponente de Hill (inhibición motora) | n_mov | [1, 4] | Agudeza de inhibición cinemática |
| Dosis máxima de infección | τ_max | Variable por virus | Dosis máxima acumulada; τᵢ = ωᵢₙ,ᵢ · τ_max |
