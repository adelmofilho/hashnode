---
title: "Simulação do dilema do Bias e da Variância"
datePublished: Mon Aug 04 2025 12:57:20 GMT+0000 (Coordinated Universal Time)
cuid: cmdx47ntl000102lb9tbb4cmq
slug: bias-variance-tradeoff-simulation
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/S-kyxdRsQP4/upload/09c0c47a7213cc4e0d3060d2659ad585.jpeg
tags: artificial-intelligence, statistics, machine-learning

---

> Todo bom livro de machine learning trata do [dilema do bias-variância](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) e utiliza sua decomposição na discussão das técnicas de validação cruzada e ajuste de hiper-parâmetros.
> 
> Para além da fundamentação teórica, apresento, neste post, um framework básico para decomposição do bias, variância e do erro irredutível independentementede um modelo preditivo qualquer. Ao final chegaremos aos famosos gráficos de decomposição do bias-variância apresentados nos livros, mas que comumente não possuem explicação quanto sua construção.

## **Revisão de estatística**

Dois conceitos são chave na teoria de probabilidades e no desenvolvimento do dilema do bias-variância: a esperança matemática (valor esperado) e a variância.

A esperança estatística é definida matematicamente como a integral em todo intervalo real do produto da variável aleatória x pela sua probabilidade de ocorrência.

$$E[X] = \int_{-\infty}^{\infty} x f(x) \, dx$$

Tratando-se de uma variável x discreta, a integral assume a forma de um somatório.

$$E[X] = \sum_{i=1}^{\infty} x_i p(x_i)$$

A análise da equação acima nos permite identificar que a esperança de uma variável discreta corresponde a sua média aritmética, uma vez que \\(p(x)\\) pode ser expresso como a razão entre o número de observações de um determinado \\(x_i\\) e o número total de observações.

A variância é definida matematicamente como a esperança do quadrado da diferença entre a variável e seu valor esperado.

$$\text{Var}(X) = E\left[(X - E(X))^2\right]$$

A expansão dos termos da equação acima, nos leva a seguinte expressão:

$$Var(X)=E[(X2−2⋅X⋅E(X)+(E(X))2)]$$

E, ao aplicarmos o operador esperança no polinômio acima e utilizando a sua propriedade cumulativa chegamos a seguinte expressão:

$$\text{Var}(X) = E\left[X^2 - 2 \cdot X \cdot E(X) + (E(X))^2\right]$$

Sendo a esperança de uma variável X uma constante, a esperança dela equivalente à própria.

$$\text{Var}(X) = E(X^2) - 2 \cdot (E(X))^2 + (E(X))^2 = E(X^2) - (E(X))^2$$

Para facilitar a decomposição do bias-variância que veremos a seguir, vamos organizar a equação acima da seguinte forma:

$$E(X^2) = \text{Var}(X) + (E(X))^2$$

## Decomposição do erro esperado

Dada uma variável \\(y^∗\\) cuja predição é desejada, iremos assumir a existência de um modelo idealizado \\(f(x)\\) capaz de explicar integralmente o comportamento desta variável.

$$y^* = f(x)$$

Na prática, \\(y^∗\\) está sujeita à sua própria variabilidade, de forma que expandimos a equação acima para a forma:

$$y = f(x) + \sigma_y$$

Apesar de desconhecermos \\(f(x)\\), é possível predizer os valores de y através de modelos matemáticos empíricos baseados em dados, o qual denominaremos \\(\hat{f}(x)\\). O objetivo de toda modelagem é obter uma função \\(\hat{f}(x)\\) cujos valores preditos sejam os mais próximos possíveis de y para um dado vetor de preditores \\(x=[x_1,x_2,…x_p]\\).

Vamos supor que para medir o quão “próximo” são os valores de \\(y\\) e \\(\hat{f}(x)\\) utilizaremos uma medida quadrática tal como o erro quadrático, isto é \\((y−\hat{f}(x_0))^2\\), neste caso, nosso erro esperado. Observe que estamos avaliando a função em um único ponto x0.

É fato que para cada conjunto de dados utilizado obteremos um diferente valor de \\(\hat{f}(x_0)\\) e, de forma equivalente, um diferente valor de \\(y\\). Para compensar essa variabilidade e obter um valor representativo podemos trabalhar com a esperança estatística da quantidade quadrática que estamos avaliando.

$$E(\epsilon) = E\left((y - \hat{f}(x_0))^2\right)$$

Expandindo a quantidade acima, chegamos a seguinte expressão.

$$E\left((y - \hat{f}(x_0))^2\right) = E\left(y^2 + (\hat{f}(x_0))^2 - 2 \cdot y \cdot \hat{f}(x_0)\right)$$

$$E\left((y - \hat{f}(x_0))^2\right) = E(y^2) + E((\hat{f}(x_0))^2) - 2 \cdot E(y \cdot \hat{f}(x_0))$$

Utilizando a transformação quanto a variância apresentada na seção 1, podemos expandir os termos com a esperança da variável ao quadrado para a forma:

$$E\left((y - \hat{f}(x_0))^2\right) = \text{Var}(y) + (E(y))^2 + \text{Var}(\hat{f}(x_0)) + (E(\hat{f}(x_0)))^2 - 2 \cdot E(y \cdot \hat{f}(x_0))$$

Rearranjamos os termos para fins de praticidade dos próximos passos:

$$E\left((y - \hat{f}(x_0))^2\right) = \text{Var}(y) + \text{Var}(\hat{f}(x_0)) + \left[(E(y))^2 + (E(\hat{f}(x_0)))^2 - 2 \cdot E(y \cdot \hat{f}(x_0))\right]$$

A expressão acima pode ter seu último termo simplificado pelo conceito de produto notável para a forma:

$$E\left((y - \hat{f}(x_0))^2\right) = \text{Var}(y) + \text{Var}(\hat{f}(x_0)) + \left[E(y) - E(\hat{f}(x_0))\right]^2$$

Dado que \\(y=f(x)+σ_y\\), a esperança de \\(y - E(y)\\) - é igual ao próprio \\(f(x)\\) visto que se espera que a esperança, o valor médio, de \\(σ_y - E(σ_y)\\) - seja igual a zero. Por outro lado, a variância de \\(y - Var(y)\\) é igual ao próprio \\(σ_y\\), uma vez que se espera que \\(f(x)\\) tenha variância nula por ser o modelo idealizado. Desta forma, o erro esperado toma a seguinte expressão:

$$E\left((y - \hat{f}(x_0))^2\right) = \sigma_y + \text{Var}(\hat{f}(x_0)) + \left[f(x) - E(\hat{f}(x_0))\right]^2$$

A expressão acima é o que a literatura indica como a decomposição em bias-variância do erro esperado.

$$E\left((y - \hat{f}(x_0))^2\right) = \text{Erro irredutível} + \text{Variância} + (\text{Bias})^2$$

## **Interpretação dos termos da decomposição**

O bias - \\([f(x)−E(\hat{f}(x_0)]\\) - corresponde ao grau de proximidade do modelo proposto ao modelo ideal - \\(f(x)\\). Observe, contudo, que não tratamos diretamente do modelo proposto \\(\hat{f}(x_0)\\), mas da esperança (média) de modelos propostos - \\(E(\hat{f}(x_0))\\).

**E o que é a esperança de um modelo?**

O termo \\(E(\hat{f}(x_0))\\) corresponde ao valor médio de diferentes modelos \\(f(x)\\) aplicados no ponto \\(x_0\\).

Por esta razão que o bias-variância é **simulado**.

Na prática teremos um único banco de dados, obteremos um determinado \\(\hat{f}(x_0)\\) e teremos uma estimativa do erro esperado.

Não é possível decompor o erro esperado em termos do bias e da variância com um único modelo construído. Por esta razão, simulamos diferentes bancos de dados para a construção de diferentes modelos e a obtenção de diferentes estimativas pontuais.

A diferença entre o valor do modelo perfeito em um ponto \\(x_0\\) e o valor médio de diferentes modelos \\(\hat{f}(x)\\) no mesmo ponto é o que matematicamente denomina-se como bias.

**E a variância?**

A variância segue o mesmo princípio, sua estimativa passa pela construção de diferentes modelos vindo de diferentes bancos de dados, a aplicação destes modelos em um ponto \\(x_0\\) e então a estimativa da variância para cada valor de \\(\hat{f}(x_0)\\).

O último termo de nossa equação - erro irredutível - é a própria variabilidade de \\(y\\) e nos fornece uma importante conclusão: Nenhum modelo pode ter erro esperado menor que a variabilidade da variável predita.

**Tá… e o onde está o tradeoff?**

Certo! Até agora, verificamos como decompor o erro esperado nas suas parcelas de bias, variância e erro irredutível. Para enxergar o dilema (tradeoff) existente, passemos para um exemplo intuitivo!

## **Noção intuitiva do dilema do bias-variância**

Digamos que possuímos o conjunto de dados apresentado na figura a seguir. Nossa variável de interesse \\(y\\) é uma função linear da preditora \\(x\\).

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pydataset import data

np.random.seed(42)
cars = data('cars')

filtered = cars[cars['dist'] < 80]
grouped = filtered.groupby('speed', as_index=False)['dist'].mean()
sampled = grouped.sample(n=10, random_state=42)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=sampled, x='speed', y='dist')
plt.xlabel("x", fontsize=12, fontweight='bold', family='serif')
plt.ylabel("y", fontsize=12, fontweight='bold', family='serif')
plt.xticks(fontsize=12, fontweight='bold', family='serif')
plt.yticks(fontsize=12, fontweight='bold', family='serif')
plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1754268132546/6ab6c6bf-ba64-4bb6-8ff2-fb149de9c86e.png align="center")

Podemos a partir do conjunto de dados acima propor diferentes modelos polinomiais, digamos que ajustaremos os modelos de primeira ordem (linear), terceira ordem e quinta ordem e nona ordem. O perfil de cada um destes modelos é apresentado a seguir.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pydataset import data

data_seed = 42
np.random.seed(data_seed)
sns.set(style="whitegrid")

df = data("cars")
df = df[df['dist'] < 80]
df_grouped = df.groupby('speed', as_index=False)['dist'].mean()
sampled = df_grouped.sample(n=10, random_state=data_seed).sort_values(by="speed")

def plot_poly(sampled_data, degree):
    X = sampled_data['speed'].values.reshape(-1, 1)
    y = sampled_data['dist'].values

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    x_range = np.linspace(0, 25, 200).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_pred = model.predict(x_range_poly)

    plt.figure(figsize=(6, 4))
    plt.scatter(sampled_data['speed'], sampled_data['dist'], color='blue')
    plt.plot(x_range, y_pred, color='red')
    plt.xlim(0, 25)
    plt.ylim(0, 70)
    plt.xlabel("x", fontsize=12, fontweight='bold', family='serif')
    plt.ylabel("y", fontsize=12, fontweight='bold', family='serif')
    plt.xticks(fontsize=12, fontweight='bold', family='serif')
    plt.yticks(fontsize=12, fontweight='bold', family='serif')
    plt.title(f'Degree {degree}', fontsize=14, fontweight='bold', family='serif')
    plt.tight_layout()
    return plt

plots = [plot_poly(sampled, deg) for deg in [1, 3, 5, 7]]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
degrees = [1, 3, 5, 7]

for ax, degree in zip(axs.ravel(), degrees):
    poly = PolynomialFeatures(degree)
    X = sampled['speed'].values.reshape(-1, 1)
    y = sampled['dist'].values
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    x_range = np.linspace(0, 25, 200).reshape(-1, 1)
    y_pred = model.predict(poly.transform(x_range))

    ax.scatter(sampled['speed'], sampled['dist'], color='blue')
    ax.plot(x_range, y_pred, color='red')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 70)
    ax.set_title(f'Degree {degree}', fontsize=12, fontweight='bold', family='serif')
    ax.set_xlabel("x", fontsize=10, fontweight='bold', family='serif')
    ax.set_ylabel("y", fontsize=10, fontweight='bold', family='serif')
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1754268206356/3a48fdd6-74a0-4f1f-ac7e-29cabe342ce4.png align="center")

Analisando os gráficos acima, vemos que conforme aumentamos o grau do polinômio, ou seja, a complexidade do modelo, obtemos uma curva mais aderente aos dados - dizemos, desta forma, que conforme aumentamos a complexidade do modelo, reduzimos o bias associado ao modelo.

Esta observação se torna mais clara, se ao invés de poucos pontos, povoássemos todo o espaço da variável preditora. Neste caso, apesar do último modelo ser de grau 7, seu comportamento seria equivalente ao de uma reta, portanto, se aproximando do modelo ideal linear.

```python
import numpy as np
import pandas as pd
from pydataset import data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

cars = data('cars')

dc = (
    cars[cars['dist'] < 80]
    .groupby('speed', as_index=False)
    .agg({'dist': 'mean'})
    .sample(n=10, random_state=42)
)

model = LinearRegression()
model.fit(dc[['speed']], dc['dist'])

x_seq = np.arange(0, 25.01, 0.01).reshape(-1, 1)
y_pred = model.predict(x_seq)

y_noisy = y_pred + np.random.normal(0, 0.2, size=x_seq.shape[0])

df_final = pd.DataFrame({
    'speed': x_seq.flatten(),
    'dist': y_noisy
}).query('dist > 0')

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_final, x='speed', y='dist', s=10)

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
poly_model.fit(df_final[['speed']], df_final['dist'])
y_poly = poly_model.predict(x_seq)

plt.plot(x_seq, y_poly, color='red')
plt.xlabel('x', fontsize=12, fontweight='bold', family='serif')
plt.ylabel('y', fontsize=12, fontweight='bold', family='serif')
plt.xticks(fontsize=12, fontweight='bold', family='serif')
plt.yticks(fontsize=12, fontweight='bold', family='serif')
plt.xlim(0, 25)
plt.ylim(0, 70)
plt.title('Polynomial Fit (Degree 7)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1754268491194/2d571e53-7e2a-4df5-be36-076f38532ccc.png align="center")

Experimentemos, agora, gerar outros dados vindo do mesmo banco de dados `cars` e observar o comportamento dos modelos polinomiais com `set.seed(43)`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pydataset import data

data_seed = 43
np.random.seed(data_seed)
sns.set(style="whitegrid")

df = data("cars")
df = df[df['dist'] < 80]
df_grouped = df.groupby('speed', as_index=False)['dist'].mean()
sampled = df_grouped.sample(n=10, random_state=data_seed).sort_values(by="speed")

def plot_poly(sampled_data, degree):
    X = sampled_data['speed'].values.reshape(-1, 1)
    y = sampled_data['dist'].values

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    x_range = np.linspace(0, 25, 200).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_pred = model.predict(x_range_poly)

    plt.figure(figsize=(6, 4))
    plt.scatter(sampled_data['speed'], sampled_data['dist'], color='blue')
    plt.plot(x_range, y_pred, color='red')
    plt.xlim(0, 25)
    plt.ylim(0, 70)
    plt.xlabel("x", fontsize=12, fontweight='bold', family='serif')
    plt.ylabel("y", fontsize=12, fontweight='bold', family='serif')
    plt.xticks(fontsize=12, fontweight='bold', family='serif')
    plt.yticks(fontsize=12, fontweight='bold', family='serif')
    plt.title(f'Degree {degree}', fontsize=14, fontweight='bold', family='serif')
    plt.tight_layout()
    return plt

plots = [plot_poly(sampled, deg) for deg in [1, 3, 5, 7]]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
degrees = [1, 3, 5, 7]

for ax, degree in zip(axs.ravel(), degrees):
    poly = PolynomialFeatures(degree)
    X = sampled['speed'].values.reshape(-1, 1)
    y = sampled['dist'].values
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    x_range = np.linspace(0, 25, 200).reshape(-1, 1)
    y_pred = model.predict(poly.transform(x_range))

    ax.scatter(sampled['speed'], sampled['dist'], color='blue')
    ax.plot(x_range, y_pred, color='red')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 70)
    ax.set_title(f'Degree {degree}', fontsize=12, fontweight='bold', family='serif')
    ax.set_xlabel("x", fontsize=10, fontweight='bold', family='serif')
    ax.set_ylabel("y", fontsize=10, fontweight='bold', family='serif')
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1754269063841/6d8b76b5-5c06-4c95-a890-a05e82a23791.png align="center")

Compare o gráfico acima com o gerado anteriormente.

Observe que conforme se aumenta o grau do polinômio (complexidade do modelo) **menos** o modelo ajustado é similar ao equivalente na primeira figura - dizemos, neste caso, que a variância entre os modelo aumenta conforme a complexidade.

Com base nesta abordagem intuitiva fica claro que o bias reduz com a complexidade do modelo e o inverso ocorre com a variância. O erro irredutível é constante, e o erro esperado corresponde a soma destes três últimos termos.

Poderíamos repetir esse processo infinitamente, obtendo em cada uma das vezes diferentes ajustes para os parâmetros do modelo e valores diferentes para as predições em um mesmo data point. E, é exatamente isso que fazemos para realizar simulações que nos entregam o formato das curvas de bias e variância.

A partir do código a seguir, realizamos a simulação de 500 observações cujo modelo ideal é uma função quadrática, e adicionamos um ruido normal aleatório nos valores ideais da sua variável predita; ajustamos os parâmetros de modelos lineares de 0 a 8 parâmetros polinomiais e com esses valores calculamos algo que na prática não seria possível: a variância da estimação dos valores preditos e o bias dos modelos.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

fun = lambda x: x**2

x_ref = np.linspace(0, 1, 11)
y_ref = fun(x_ref)
ref_df = pd.DataFrame({'x': x_ref, 'y': y_ref})

def data_sim(n=50):
    x = np.random.uniform(0, 1, n)
    y = fun(x) + np.random.normal(0, 0.2, n)
    return pd.DataFrame({'x': x, 'y': y})

results = []

for i in range(500):
    sim = data_sim()
    eps = np.random.normal(0, 0.2, len(x_ref))
    y_obs = y_ref + eps

    for degree in range(9):
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(sim[['x']])
        model = LinearRegression().fit(X_poly, sim['y'])

        X_pred = poly.transform(ref_df[['x']])
        pred = model.predict(X_pred)

        for x_val, y_true, y_hat in zip(x_ref, y_obs, pred):
            results.append({
                'model': degree,
                'x': x_val,
                'y': y_true,
                'pred': y_hat
            })

df = pd.DataFrame(results)

var_df = df.groupby(['model', 'x'])['pred'].var().reset_index(name='varfx')
var_summary = var_df.groupby('model')['varfx'].mean().reset_index(name='varf')

mean_pred = df.groupby(['model', 'x'])['pred'].mean().reset_index(name='meanf')
mean_pred['bias2'] = (mean_pred['meanf'] - fun(mean_pred['x']))**2
bias_summary = mean_pred.groupby('model')['bias2'].mean().reset_index(name='bias2f')

df['error'] = (df['pred'] - df['y'])**2
mse_summary = df.groupby('model')['error'].mean().reset_index(name='meanmse')

summary = (
    bias_summary
    .merge(var_summary, on='model')
    .merge(mse_summary, on='model')
)
summary['eps'] = summary['meanmse'] - summary['bias2f'] - summary['varf']

summary_melted = summary.melt(id_vars='model', 
                               value_vars=['bias2f', 'varf', 'eps', 'meanmse'])

plt.figure(figsize=(10, 6))
for var in summary_melted['variable'].unique():
    subset = summary_melted[summary_melted['variable'] == var]
    plt.plot(subset['model'], subset['value'], label=var)
    plt.scatter(subset['model'], subset['value'])

plt.xlabel("Polynomial Degree", fontsize=12, fontweight='bold', family='serif')
plt.ylabel("Value", fontsize=12, fontweight='bold', family='serif')
plt.title("Bias-Variance Tradeoff", fontsize=14, fontweight='bold', family='serif')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1754268635033/b64c2c33-83f6-4e12-9a64-b19041b6598d.png align="center")

O gráfico acima resume o resultado da simulação. Como é possível verificar existe um ponto de mínimo na curva de essa médio quadrático esperado (meansmse), esse erro nunca pode ser zero (pelo ruido aleatório que adicionamos e que existirá em qualquer medição na vida real) e que se aproxima no seu limite no valor do ruído (eps). Também, conforme o grau do polinômio se aumenta, observamos uma redução do bias (bias2f), mas que por consequência eleva a variância das predições (varf).

Desta forma, conseguimos enxergar que em machine learning sempre estamos lutando por uma otimização que nunca será zero, o erro sempre existirá, cabe a nos minimizarmos ao máximo.

Até uma próxima!