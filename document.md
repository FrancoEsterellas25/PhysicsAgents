# Modelo Epidemiológico SEIRS-D
## Arquitectura Matemática Unificada
### Enfoques Discreto y Continuo con Núcleo Biológico Común (SDE-OU)

---

## Resumen de Arquitectura

La dualidad metodológica comparte un núcleo biológico común regido por Ecuaciones Diferenciales Estocásticas (SDE-OU), pero difieren en su mecanismo de contagio.

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

El núcleo biológico es de carácter transversal a las formulaciones. Los agentes no son receptores pasivos sino sistemas biológicos con umbrales de respuesta individuales.

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

## Parte VI: Análisis de Datos y Métricas de Inferencia

El enfoque orientado a datos (*Data-Oriented*) permite tratar la simulación basada en agentes como un experimento biológico clínico. Se aplican técnicas de bioestadística y análisis de sensibilidad global para extraer conclusiones robustas sobre la propagación y la letalidad.

### 1. Estimación del Ritmo Básico de Reproducción ($R_0$) y Tasa $r$

En las fases tempranas de la epidemia (donde la población susceptible $S(t) \approx N$), el crecimiento de la fracción de infectados sigue un régimen exponencial:
$$I(t) = I_0 \cdot e^{r \cdot t}$$

La tasa de crecimiento exponencial diaria ($r$) se estima mediante regresión lineal por mínimos cuadrados ordinarios (OLS) sobre el logaritmo de las infecciones activas durante la ventana inicial de crecimiento:
$$\ln(I(t)) = \ln(I_0) + r \cdot t$$

Una vez obtenida $r$, se deduce el $R_0$ exacto a través de la ecuación de Euler-Lotka simplificada para un tiempo de generación medio $T_g$ (periodo de incubación medio más el tiempo promedio hasta el aclaramiento viral):
$$R_0 = 1 + r \cdot T_g$$

---

### 2. Modelo de Cox de Riesgos Proporcionales (Time-Varying)

Para cuantificar cómo influyen el entorno espacial y la genética individual en la tasa instantánea de contagio, se ajusta un Modelo de Regresión de Cox con covariables dependientes del tiempo:
$$h_i(t) = h_0(t) \cdot \exp\!\Big(\beta_1 \cdot D_i(t) + \beta_2 \cdot \omega_{in,i}\Big)$$

* **Función de Riesgo ($h_i(t)$):** Representa la tasa instantánea de infección para el agente $i$ en el tiempo $t$, dado que ha sobrevivido como susceptible hasta ese instante.
* **Riesgo Basal ($h_0(t)$):** El riesgo compartido cuando todas las covariables son nulas.
* **Covariables:**
  * **$D_i(t)$ (Dosis local de aerosol):** Covariable dinámica que cambia en cada paso temporal según la cercanía a focos infecciosos.
  * **$\omega_{in,i}$ (Inmunidad innata basal):** Covariable estática determinada genéticamente al inicio por la cópula.

#### Interpretación de Hazard Ratios (HR):
* **$\text{HR}_{\text{dosis}} = e^{\beta_1}$:** Multiplicador del riesgo instantáneo de contagio por cada desviación estándar de incremento en la dosis de aerosol respirada. Un $\text{HR} > 1.0$ valida físicamente que la cercanía espacial a las nubes de aerosol eleva el riesgo de contagio.
* **$\text{HR}_{\omega_{in}} = e^{\beta_2}$:** Multiplicador del riesgo por cada desviación estándar de aumento en la inmunidad innata. Un $\text{HR} < 1.0$ confirma la efectividad protectora de la barrera inmunológica del agente.

---

### 3. Curvas de Supervivencia de Kaplan-Meier

El tiempo de permanencia en el compartimento susceptible antes de ser contagiado ($T_i = \tau_i^{inf}$) se modela como un análisis de supervivencia clásica. La probabilidad de que un agente permanezca sano más allá del tiempo $t$ se calcula mediante el estimador de Kaplan-Meier:
$$\hat{S}(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)$$

* **$d_j$:** Número de nuevos agentes que transitan de $S \to E$ (eventos de infección) en el paso de tiempo $t_j$.
* **$n_j$:** Población en riesgo inmediatamente antes de $t_j$ (agentes que continúan en estado susceptible $S$).
* **Censura Derecha:** Los agentes que finalizan la simulación sin haberse contagiado se marcan como observaciones censuradas en $T = t_{max}$, aportando información al denominador $n_j$ hasta el final de la serie sin contar como eventos.

El análisis se expone estratificado dividiendo la población según la mediana de la inmunidad innata ($\omega_{in}$), permitiendo evaluar la separación de curvas de supervivencia y demostrar el impacto de la heterogeneidad genética.

---

### 4. Análisis de Sensibilidad Global de Sobol (Método Jansen)

Para evaluar qué parámetros del simulador dominan la varianza de la métrica final (Tasa de Mortalidad $Y = D(t_{final})/N$), se realiza un análisis de sensibilidad global basado en la descomposición de la varianza de Sobol:
$$V(Y) = \sum_{i} V_i + \sum_{i < j} V_{ij} + \dots + V_{12\dots k}$$

#### 4.1 Índice de Primer Orden (Efecto Directo $S_i$)
Mide la fracción de la varianza de la mortalidad que se atribuye exclusivamente a la variación del parámetro $X_i$ de forma aislada:
$$S_i = \frac{V_i}{V(Y)} = \frac{V_{X_i}\big(E_{\mathbf{X}_{\sim i}}[Y|X_i]\big)}{V(Y)}$$

#### 4.2 Índice Total (Efecto Total $S_{Ti}$)
Mide la contribución total del parámetro $X_i$ a la varianza de la mortalidad, incluyendo su efecto directo y todas sus interacciones no lineales con los demás parámetros:
$$S_{Ti} = \frac{E_{\mathbf{X}_{\sim i}}\big[V_{X_i}(Y|\mathbf{X}_{\sim i})\big]{}}{V(Y)} = 1 - \frac{V_{\mathbf{X}_{\sim i}}\big(E_{X_i}[Y|\mathbf{X}_{\sim i}]\big)}{V(Y)}$$

#### 4.3 Estimador Robusto de Jansen (1999)
Para calcular estos índices de manera eficiente y numéricamente estable ante modelos altamente estocásticos, el script genera matrices de Saltelli $A, B \in \mathbb{R}^{N \times k}$ y matrices mixtas $C^{(i)}$ donde todas las columnas provienen de $B$ excepto la columna $i$ que proviene de $A$.

Los estimadores de Jansen calculan la varianza condicional a partir de las diferencias cuadráticas de las respuestas del modelo ($Y_A, Y_B, Y_{C^{(i)}}$):
* **Varianza Total:**
  $$V(Y) = \text{Var}(Y_A \cup Y_B)$$
* **Efecto Total ($S_{Ti}$):**
  $$S_{Ti} = \frac{\frac{1}{2N} \sum_{j=1}^{N} (Y_{A, j} - Y_{C^{(i)}, j})^2}{V(Y)}$$
* **Efecto Directo ($S_i$):**
  $$S_i = 1 - \frac{\frac{1}{2N} \sum_{j=1}^{N} (Y_{B, j} - Y_{C^{(i)}, j})^2}{V(Y)}$$

*Nota de calibración:* Para evitar que el ruido estocástico del movimiento y las SDE contamine los estimadores de Sobol, la semilla estocástica interna de la simulación se congela uniformemente en cada corrida del estimador.

---

### 5. Dataset Multidimensional (Panel Data)

Cada simulación exporta un conjunto de datos en formato panel con series temporales para cada agente $i$ a lo largo de pasos de tiempo $t$:
* **Coordenadas espaciales:** $\{x_i(t), y_i(t)\}$ (enfoque continuo).
* **Dinámica viral:** $\{v_i(t)\}$ (carga viral individual).
* **Contacto y patología:** $\{D_i(t), \text{AUC}_i(t)\}$ (inóculo acumulado y daño).
* **Estado epidemiológico:** $\text{Estado}_i(t) \in \{S, E, I, R, D\}$.
* **Metadatos estáticos del agente:** $[\omega_{in,i}, \omega_{ad,i}, \text{edad}_i]$.

Este dataset estructurado en parquet permite realizar auditorías de datos secundarias y calibración de modelos epidemiológicos empíricos externos.

---

## Parte VII: Restricciones y Límites Analíticos

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

### 6. Cálculo del AUCᵢ y Escala del Argumento Logístico (Transversal)

El argumento de la función logística (AUCᵢ − ωₐd,ᵢ) mezcla magnitudes de distinta escala: AUCᵢ crece con la duración de la infección (potencialmente >> 1) mientras que ωₐd,ᵢ ∈ [0, 1]. Es imprescindible normalizar AUCᵢ al rango unitario — por ejemplo, dividiendo por AUC_max esperado bajo el patógeno simulado — antes de calibrar λ, o absorber la escala en λ de forma explícita durante la calibración.



---

## Parte VIII: Protocolo de Inicialización y Setup de Agentes

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

## Parte IX: Medidas de Intervención, Protección e Higiene

El sistema de modelado permite evaluar el impacto de políticas de salud pública (intervenciones colectivas) y cambios de conducta (protección personal e higiene) sobre la dinámica de propagación del virus.

### 1. Medidas de Intervención Colectiva (Gatillos de Contención)

Ambos entornos (discreto y continuo) evalúan dinámicamente la prevalencia del virus $P(t) = \frac{|I(t)|}{N}$ para activar de forma adaptativa las medidas de contención cuando se supera un umbral crítico.

#### 1.1 Cuarentena Doméstica y Aislamiento de Infectados
* **Objetivo:** Restringir el movimiento y los contactos de los individuos infecciosos/expuestos para detener la propagación del virus.
* **Enfoque Continuo:** Si la prevalencia supera el gatillo de cuarentena ($P(t) \ge \text{gatillo\_cuarentena}$), todos los agentes expuestos ($E$) o infectados ($I$) ven reducida su difusividad espacial a cero:
  $$D_{esp}(V_i) = 0 \qquad \forall i \in \{E, I\}$$
  Físicamente, esto confina a los agentes enfermos en sus posiciones geográficas actuales (aislamiento en el hogar), evitando que propaguen aerosoles por el mapa.
* **Enfoque Discreto:** Si el gatillo de cuarentena se activa, se asume un aislamiento domiciliario con un 95% de efectividad. La carga viral efectiva que una celda enferma emite a sus celdas vecinas se escala por un factor de atenuación:
  $$V_{efectiva, j} = 0.05 \cdot V_j \qquad \forall j \in \{I, E\}$$

#### 1.2 Distanciamiento Social
* **Objetivo:** Incrementar la separación física promedio entre agentes en la sociedad.
* **Enfoque Continuo:** Al activarse el gatillo, el radio efectivo de la pluma de aerosol ($\ell$) se reduce linealmente de acuerdo con el cumplimiento del distanciamiento ($C_{DS} \in [0, 1]$):
  $$\ell_{efectivo} = \ell \cdot (1.0 - 0.5 \cdot C_{DS})$$
  Esto reduce el volumen del campo de inóculo alrededor de los infectados, requiriendo que los susceptibles se acerquen físicamente mucho más para acumular dosis de contagio.
* **Enfoque Discreto:** Al activarse, la probabilidad de transmisión sigmoide de Hill entre celdas contiguas se mitiga directamente según el nivel de acatamiento:
  $$\sigma_{efectiva} = \sigma(V_j, \omega_{in,i}) \cdot (1.0 - C_{DS})$$

---

### 2. Medidas de Protección Personal e Higiene

#### 2.1 Uso de Mascarillas (Barbijo)
El uso de mascarillas se define por un nivel de adopción en la población $P_{mask} \in [0, 1]$ y dos eficiencias de filtración:
* **Eficiencia de Emisión ($\eta_{em}$):** Fracción de partículas virales que la máscara retiene al exhalar.
* **Eficiencia de Recepción ($\eta_{rec}$):** Fracción de virus filtrada al inhalar del ambiente.

En el **Enfoque Continuo**, cada agente $i$ recibe un estado binario de mascarilla $M_i \in \{0, 1\}$ extraído de $P_{mask}$. La filtración actúa en dos etapas físicas:
1. **Emisión (Origen):** La emisión de carga viral al aire por parte de un infectado $j$ se mitiga por su máscara:
   $$I_{j,t} = I_{j,t}^{basal} \cdot (1.0 - \eta_{em} \cdot M_j)$$
2. **Recepción (Destino):** La dosis inhalada por un susceptible $i$ se filtra antes de entrar a su torrente:
   $$\Delta D_i(t) = \Delta D_i^{basal}(t) \cdot (1.0 - \eta_{rec} \cdot M_i)$$

#### 2.2 Higiene Personal (Lavado de Manos y Sanitización)
El parámetro de higiene personal $\eta_{hig} \in [0, 1]$ representa el aumento de la barrera de bioseguridad del huésped ante el patógeno (ej: desactivación de virus en superficies antes de ingresar a vías respiratorias):
* **Enfoque Continuo:** Eleva directamente el umbral de tolerancia acumulada de dosis necesario para infectarse:
  $$\tau_i = \omega_{in,i} \cdot \tau_{max} \cdot (1.0 + \eta_{hig})$$
* **Enfoque Discreto:** Eleva el umbral de carga viral de saturación $V_{50}$ de la celda receptora, dificultando la transmisión en cada paso de tiempo:
  $$V_{50}(\omega_{in,i}) = V_{50,basal} \cdot (1 + \gamma \cdot \omega_{in,i}) \cdot (1.0 + \eta_{hig})$$

---

### 3. Catálogo de Patógenos y Modelado Conceptual

Para garantizar consistencia biológica, las características clínicas e infectivas intrínsecas de cada patógeno se definen bajo la estructura inmutable `VirusProfile`. Al seleccionar un virus del catálogo, todas sus constantes físicas y de letalidad se inyectan en los simuladores. 

A continuación se detalla la descripción conceptual y la parametrización de los patógenos modelados:

#### 3.1 Sarampión (Measles)
* **Descripción Conceptual:** Es la enfermedad viral de máxima contagiosidad humana por vía aérea. Se transmite a través de aerosoles extremadamente finos ($<5\mu\text{m}$) que quedan suspendidos en el aire y viajan largas distancias. Su dosis infectiva mínima es minúscula y genera inmunidad de por vida.
* **Ecuaciones y Parámetros:**
  * **Radio de Aerosol ($\ell = 4.0$ m):** Define una pluma de propagación espacial de largo alcance.
  * **Decaimiento Lento ($\delta_{cerrado} = 0.15$):** Refleja la estabilidad del virus en interiores, permaneciendo viable más de 2 horas en aire estancado.
  * **Tolerancia Mínima ($\tau_{max} = 2.0$):** El umbral infeccioso es bajísimo, garantizando contagio casi inmediato al cruzarse con la pluma de aerosol.
  * **Inmunidad Permanente ($\mu_R = 36500$):** Recuperados permanecen en $R$ de por vida (inmunidad adquirida robusta).

#### 3.2 COVID-19 Delta (SARS-CoV-2 B.1.617.2)
* **Descripción Conceptual:** Variante caracterizada por una alta replicación viral y transmisión predominantemente aérea por aerosoles mixtos. Presenta fenómenos de superpropagación en espacios cerrados mal ventilados y pérdida de inmunidad acelerada (waning).
* **Ecuaciones y Parámetros:**
  * **Radio de Aerosol ($\ell = 2.5$ m):** Alcance espacial intermedio.
  * **Decaimiento en Interiores ($\delta_{cerrado} = 0.5$):** Mayor viabilidad y persistencia en interiores en comparación con el exterior ($\delta_{ext} = 1.5$).
  * **Tasa de Letalidad Moderada ($\lambda = 3.5$):** Curva logística de muerte sensible al daño acumulado por carga viral (AUC).
  * **Pérdida de Inmunidad Corta ($\mu_R = 120$):** Simula el decaimiento de anticuerpos a los 4 meses post-infección.

#### 3.3 Influenza Estacional (H3N2/H1N1)
* **Descripción Conceptual:** Gripe común autolimitada. Se transmite principalmente a través de gotas grandes de saliva exhaladas al toser o estornudar, limitando su alcance espacial directo. Requiere una alta dosis de exposición y el virus es muy lábil a condiciones ambientales externas.
* **Ecuaciones y Parámetros:**
  * **Radio de Aerosol Corto ($\ell = 1.2$ m):** El contagio se acota al contacto cercano (gotas pesadas).
  * **Inactivación Rápida ($\delta_{ext} = 4.0$):** El virus se inactiva rápidamente al salir del huésped, especialmente en exteriores ($\delta_{abierto} = 8.0$).
  * **Tolerancia Elevada ($\tau_{max} = 15.0$):** Se requiere inhalar una cantidad masiva de viriones para superar la inmunidad de barrera inicial.

#### 3.4 Gripe Pandémica 2009 (H1N1)
* **Descripción Conceptual:** Cepa pandémica con un crecimiento explosivo inicial debido a la falta de inmunidad previa en población joven. Transmisión moderada y letalidad general baja, pero con propagación comunitaria veloz.
* **Ecuaciones y Parámetros:**
  * **Incubación Corta ($k_E = 2, p_E = 0.5$):** Periodo de latencia promedio de solo 2 días, facilitando un crecimiento exponencial acelerado de la curva epidemiológica.
  * **Baja Letalidad ($\lambda = 0.2$):** Probabilidad de transición a fallecido ($D$) muy reducida.

#### 3.5 Ébola (Ebolavirus cepa Zaire)
* **Descripción Conceptual:** Fiebre hemorrágica viral de extrema letalidad. No se transmite por el aire en condiciones naturales; el contagio requiere contacto físico directo con fluidos corporales infectados (sangre, sudor, etc.).
* **Ecuaciones y Parámetros:**
  * **Radio de Aerosol Mínimo ($\ell = 0.25$ m):** Modela la transmisión exclusivamente por contacto directo o salpicadura en proximidad absoluta.
  * **Letalidad Extrema ($\lambda = 12.0$):** La pendiente logística es masiva, disparando la mortalidad del agente enfermo a un rango del 50% al 90% según su inmunidad adaptativa.
  * **Tolerancia Alta ($\tau_{max} = 28.0$):** Refleja la barrera de contagio en ausencia de contacto húmedo directo.

#### 3.6 Tuberculosis (Mycobacterium tuberculosis DS)
* **Descripción Conceptual:** Infección bacteriana crónica de progresión lenta y alta persistencia ambiental. Se propaga por núcleos goticulares finos que resisten la luz solar y la desecación durante horas o días.
* **Ecuaciones y Parámetros:**
  * **Persistencia Extrema ($\delta_{cerrado} = 0.05$):** La bacteria permanece suspendida y viable en espacios cerrados sin ventilación por periodos prolongados.
  * **Incubación de Larga Latencia ($k_E = 4, p_E = 0.12$):** Periodo promedio en estado expuesto ($E$) de ~29 días (simulando la lenta transición a tuberculosis pulmonar activa).
  * **Alta Resistencia Exterior ($\delta_{ext} = 0.08$):** La bacteria sobrevive significativamente fuera del huésped.

#### 3.7 VIH/SIDA
* **Descripción Conceptual:** Infección viral crónica caracterizada por el debilitamiento progresivo y permanente del sistema inmune (destrucción de linfocitos T CD4+). No existe recuperación natural y el contagio es únicamente por fluidos.
* **Ecuaciones y Parámetros:**
  * **Sin Recuperación ($\mu_R = 0$ y $M_R = 0$):** Transición a recuperado desactivada.
  * **Latencia de Años ($k_E = 10, p_E = 0.001$):** Incubación promedio extremadamente larga para simular la fase de latencia asintomática de años antes de manifestarse la enfermedad.
  * **Contacto Directo ($\ell = 0.0$):** Sin dispersión aérea alguna.

#### 3.8 SARS-CoV-1 (2003)
* **Descripción Conceptual:** Coronavirus con alta patogenicidad y letalidad. Se asocia fuertemente con eventos de superpropagación en entornos hospitalarios (transmisión nosocomial), facilitando su contención epidemiológica temprana mediante el aislamiento riguroso de casos clínicos evidentes.
* **Ecuaciones y Parámetros:**
  * **Radio Intermedio ($\ell = 2.0$ m):** Transmisión por proximidad física estrecha.
  * **Letalidad Alta ($\lambda = 6.0$):** CFR global de referencia superior al 9%, provocando un colapso clínico rápido.

---

## Apéndice: Tabla de Parámetros

### Parámetros del Núcleo Biológico

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
