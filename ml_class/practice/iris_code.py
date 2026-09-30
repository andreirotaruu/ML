# %% [markdown]
# # Assignment 1 

# %% [markdown]
# # Import the necessary modules and dataset

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ### load the dataframe

# %%
column_names = [
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width",
    "Species"
]

iris_df = pd.read_csv("iris.data", names=column_names)

print(iris_df)

# %% [markdown]
# ## 1. Compute the standard deviation and boxplots

# %%
#ignore the species column
iris_attributes = iris_df[
    ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
]
print(iris_attributes.std())

# %% [markdown]
# ### Create boxplots

# %%
iris_attributes.boxplot()

plt.title("Boxplots of Iris Attributes")
plt.ylabel("Measurement (cm)")
plt.show()

# %% [markdown]
# ## 2. Generate Histograms 

# %%
iris_attributes.hist(bins=10)

plt.show()

# %% [markdown]
# ## 3. Compute QQ Plot between Sepal.Length and Sepal.Width

# %%
#get sepal length and width and sort values to compare small values with small values
sepal_length = iris_df["Sepal.Length"].sort_values()
sepal_width = iris_df["Sepal.Width"].sort_values()

#create scatter
plt.scatter(sepal_length, sepal_width)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("QQ Plot: Sepal Length vs Sepal Width")

plt.show()

# %% [markdown]
# ## 4. Compute Scatter Plots between Sepal.Length and Sepal.Width, and Sepal.Length and Petal.Length

# %% [markdown]
# ### Sepal.Length and Sepal.Width

# %%
plt.scatter(iris_df["Sepal.Length"], iris_df["Sepal.Width"])

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")

plt.show()

correlation = iris_df["Sepal.Length"].corr(iris_df["Sepal.Width"])

print(f'Correlation: {correlation}')

# %% [markdown]
# ### Sepal Length and Petal Length

# %%
plt.scatter(iris_df["Sepal.Length"], iris_df["Petal.Length"])

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")

plt.show()

correlation = iris_df["Sepal.Length"].corr(iris_df["Petal.Length"])

print(f'Correlation{correlation}')


