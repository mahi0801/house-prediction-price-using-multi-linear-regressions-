```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv(r"C:\Users\Administrator\Downloads\sample file.csv")
df
df.bedrooms.median()
```




    3.0




```python
import math 
df1 = math.floor(df.bedrooms.median())
df1
```




    3




```python
df.bedrooms=df.bedrooms.fillna(df1)
df.bedrooms
```




    0    3.0
    1    4.0
    2    3.0
    3    3.0
    Name: bedrooms, dtype: float64




```python
df.bedrooms
```




    0     3.0
    1     4.0
    2     3.0
    3     5.0
    4     3.0
         ... 
    94    3.0
    95    3.0
    96    3.0
    97    3.0
    98    3.0
    Name: bedrooms, Length: 99, dtype: float64




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3.0</td>
      <td>20</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4.0</td>
      <td>15</td>
      <td>64000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>3.0</td>
      <td>18</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3.0</td>
      <td>30</td>
      <td>85000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>bedrooms</th>
      <th>age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2600</td>
      <td>3.0</td>
      <td>20</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000</td>
      <td>4.0</td>
      <td>15</td>
      <td>64000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3200</td>
      <td>3.0</td>
      <td>18</td>
      <td>72000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3600</td>
      <td>3.0</td>
      <td>30</td>
      <td>85000</td>
    </tr>
  </tbody>
</table>
</div>




```python
reg = linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]],df.price)
```




    LinearRegression()




```python
reg.coef_
```




    array([   28.75, -1875.  ,   125.  ])




```python
reg.intercept_
```




    -16624.999999999956




```python
reg.predict([[3000,4,15]])
```

    C:\Users\Administrator\anaconda3\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    




    array([64000.])




```python
#mutilinear
#y=m1*x1+m2*x2+m3*x3+c
#m is coefficient
#c is intercept
28.75*3600+-1875*3+125*30+-16624.999999999956
```




    85000.00000000004




```python
# finally we pridict the house price , 85000.00000000004

```


```python

```
