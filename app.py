import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio
import extra_graphs
import streamlit as st

st.title("Proyek Akhir Pengantar Data Science")
st.markdown("Welcome to Last Project in Introduction Data Science we use [Streamlit](www.streamlit.io)! For this exercise, we'll use an [dataset](https://github.com/ugis22/clustering_analysis/blob/master/clustering_analysis.ipynb).")
st.markdown("> 1. Nelson Alfons Abilo \n >2. Billy Kurniananda\n >3. Hanif Darmawan \n >4. Rakha Hanif \n")
st.header("First Step")
st.markdown("> For the first step, we can download from the data set used. The problems in this dataset are related to grouping clients or customers based on certain product data. \n\n Ugis22")
st.header("Read Data into a Dataframe")
customers = pd.read_csv("customers.csv")
customers2 = pd.read_csv("customers.csv")


#rename cs2
df = customers2.rename(columns={'CustomerID': 'CustomerID', 'Gender': 'Gender', 'Age' : 'Age', 'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending' })

st.markdown("The following is a display of the top 5 data in the customer dataset:")
st.dataframe(customers.head())

st.subheader("Selecting a subset of columns")
st.write("Here is a column subset where we can see which columns we want to display")
defaultcols = ["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]
cols = st.multiselect("Columns", customers.columns.tolist(), default=defaultcols)
st.dataframe(customers[cols].head(8))

st.header("Average Annual Income by Gender")
st.write("Here is a Average Annual Income by Gender")
st.table(df.groupby("Gender").Income.mean().reset_index()\
    .round(2).sort_values("Income", ascending=False)\
    .assign(avg_annual_income=lambda x: x.pop("Income").apply(lambda y: "%.2f" % y)))

st.header("Average Spending Score by Gender")
st.write("Here is a Spending Score by Gender")
st.table(df.groupby("Gender").Spending.mean().reset_index()\
    .round(2).sort_values("Spending", ascending=False)\
    .assign(avg_spending_score=lambda x: x.pop("Spending").apply(lambda y: "%.2f" % y)))

st.header("Exploring the data")

st.markdown("**Missing values in each variable:**")
nulls = customers.isnull().sum()
cols = customers.columns.values.tolist()
for a,b in zip(nulls, cols):
    st.markdown(f"{b} : {a}")

st.markdown(f"**Duplicated rows:** {customers.duplicated().sum()}")

st.markdown("**Columns type:**")
c_dtypes = customers.dtypes
for a,b in zip(c_dtypes, cols):
    st.markdown(f"{b} : {a}")


st.header("The Distibution of Annual Income?")
st.write("""Select a custom annual income range from the side bar to update the histogram below displayed as a Plotly chart using
[`st.plotly_chart`](https://streamlit.io/docs/api.html#streamlit.plotly_chart).""")
values = st.sidebar.slider("Annual Income range", float(df.Income.min()), float(df.Income.clip(upper=1000.).max()), (0., 150.))
f = px.histogram(df.query(f"Income.between{values}"), x="Income", nbins=15, title="Annual Income distribution")
f.update_xaxes(title="Income")
f.update_yaxes(title="Frequency")
st.plotly_chart(f)

st.header("Annual Income by number of age")
st.write("Enter a range of age in the sidebar to view annual income in that range.")
minimum = st.sidebar.number_input("Minimum", min_value=0, value =17)
maximum = st.sidebar.number_input("Maximum", min_value=0)
if minimum > maximum:
    st.error("Please enter a valid range")
else:
    df.query("@minimum<=Age<=@maximum").sort_values("Age", ascending=False)\
        .head(50)[["CustomerID", "Gender", "Income", "Spending"]]


def statistics(variable):
    if variable.dtype == "int64" or variable.dtype == "float64":
        return pd.DataFrame([[variable.name, np.mean(variable), np.std(variable), np.median(variable), np.var(variable)]], 
                            columns = ["Variable", "Mean", "Standard Deviation", "Median", "Variance"]).set_index("Variable")
    else:
        return pd.DataFrame(variable.value_counts())

def graph_histo(x):
    if x.dtype == "int64" or x.dtype == "float64":
        # Select size of bins by getting maximum and minimum and divide the substraction by 10
        size_bins = 10
        # Get the title by getting the name of the column
        title = x.name
        #Assign random colors to each graph
        color_kde = list(map(float, np.random.rand(3,)))
        color_bar = list(map(float, np.random.rand(3,)))

        # Plot the displot
        sns.distplot(x, bins=size_bins, kde_kws={"lw": 1.5, "alpha":0.8, "color":color_kde},
                       hist_kws={"linewidth": 1.5, "edgecolor": "grey",
                                "alpha": 0.4, "color":color_bar})
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.ylabel("Frequency", size=16, labelpad=15)
        # Customize title
        plt.title(title, size=18)
        # Customize grid and axes visibility
        plt.grid(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
    else:
        x = pd.DataFrame(x)
        # Plot       
        sns.catplot(x=x.columns[0], kind="count", palette="spring", data=x)
        # Customize title
        title = x.columns[0]
        plt.title(title, size=18)
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.xlabel("")
        plt.ylabel("Counts", size=16, labelpad=15)        
        # Customize grid and axes visibility
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("**Spending score:**")
spending = customers["Spending Score (1-100)"]
st.dataframe(statistics(spending).head())

st.pyplot(graph_histo(spending))

age = customers["Age"]
st.dataframe(statistics(age).head())

st.pyplot(graph_histo(age))

income = customers["Annual Income (k$)"]
st.dataframe(statistics(income).head())

st.pyplot(graph_histo(income))

gender = customers["Gender"]
st.dataframe(statistics(gender).head())

st.pyplot(graph_histo(gender))

st.subheader("Correlation between parameters")
st.pyplot(sns.pairplot(customers, x_vars = ["Age", "Annual Income (k$)", "Spending Score (1-100)"], 
               y_vars = ["Age", "Annual Income (k$)", "Spending Score (1-100)"], 
               hue = "Gender", 
               kind= "scatter",
               palette = "YlGnBu",
               height = 2,
               plot_kws={"s": 35, "alpha": 0.8}))

st.header("Dimensionality reduction")
st.subheader("Principal Component Analysis (PCA)")
customers["Male"] = customers.Gender.apply(lambda x: 0 if x == "Male" else 1)
customers["Female"] = customers.Gender.apply(lambda x: 0 if x == "Female" else 1)
X = customers.iloc[:, 2:]
st.dataframe(X.head())
st.markdown("In order to apply PCA, we are going to use the PCA function from sklearn module.")
pca = PCA(n_components=2).fit(X)
st.markdown("During the fitting process, the model learns some quantities from the data: the 'components' and 'explained variance'.")
st.markdown(pca.components_)
st.markdown(pca.explained_variance_)

pca_2d = pca.transform(X)
st.pyplot(extra_graphs.biplot(pca_2d[:,0:2], np.transpose(pca.components_[0:2, :]), labels=X.columns))

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss, c="#c51b7d")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title('Elbow Method', size=14)
plt.xlabel('Number of clusters', size=12)
plt.ylabel('wcss', size=14)
st.pyplot(plt.show())

st.markdown("How does k-means clustering works? The main idea is to select k centers, one for each cluster. There are several ways to initialize those centers. We can do it randomly, pass certain points that we believe are the center or place them in a smart way (e.g. as far away from each other as possible). Then, we calculate the Euclidean distance between each point and the cluster centers. We assign the points to the cluster center where the distance is minimum. After that, we recalculate the new cluster center. We select the point that is in the middle of each cluster as the new center.  And we start again, calculate distance, assign to cluster, calculate new centers. When do we stop? When the centers do not move anymore.")
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=10, n_init=10, random_state=0)
y_means = kmeans.fit_predict(X)

fig, ax = plt.subplots(figsize = (8, 6))
st.markdown("Now, let's check how our clusters look like:")
plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
            c=y_means, 
            edgecolor="none", 
            cmap=plt.cm.get_cmap("Spectral_r", 5),
            alpha=0.5)
        
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)

plt.xticks(size=12)
plt.yticks(size=12)

plt.xlabel("Component 1", size = 14, labelpad=10)
plt.ylabel("Component 2", size = 14, labelpad=10)

plt.title('Dominios agrupados en 5 clusters', size=16)


plt.colorbar(ticks=[0, 1, 2, 3, 4])

st.pyplot(plt.show())

centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ["Age", "Annual Income", "Spending", "Male", "Female"])
centroids.index_name = "ClusterID"
centroids["ClusterID"] = centroids.index
centroids = centroids.reset_index(drop=True)
st.dataframe(centroids)

st.markdown("The most important features appear to be Annual Income and Spending score.  We have people whose income is low but spend in the same range - segment 0. People whose earnings a high and spend a lot - segment 1. Customers whose income is middle range but also spend at the same level - segment 2.  Then we have customers whose income is very high but they have most spendings - segment 4. And last, people whose earnings are little but they spend a lot- segment 5.Imagine that tomorrow we have a new member. And we want to know which segment that person belongs. We can predict this.")
X_new = np.array([[43, 76, 56, 0, 1]]) 
 
new_customer = kmeans.predict(X_new)
st.markdown(f"***The new customer belongs to segment {new_customer[0]}***")