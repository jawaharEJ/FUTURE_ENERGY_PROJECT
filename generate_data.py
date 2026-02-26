import pandas as pd
import numpy as np

# Generate realistic India-based energy data
years = list(range(2015, 2025))
months = list(range(1, 13))

data = []

population_base = 1.3e9  # Starting population in 2015
growth_rate = 0.015  # 1.5% annual growth

for year in years:
    pop = population_base * (1 + growth_rate) ** (year - 2015)
    for month in months:
        industrial_growth = np.random.uniform(4.0, 8.0)  # Random industrial growth 4-8%
        # Energy consumption roughly proportional to population and growth
        base_consumption = 120000  # Base monthly consumption in GWh
        consumption = base_consumption * (pop / 1.3e9) * (1 + industrial_growth / 100) * (1 + (month - 6) * 0.02)  # Seasonal variation
        data.append({
            'Year': year,
            'Month': month,
            'Population': pop,
            'Industrial_Growth': industrial_growth,
            'Energy_Consumption': consumption
        })

df = pd.DataFrame(data)
df.to_csv('energy_data.csv', index=False)
print("New energy_data.csv generated with India-based realistic data.")