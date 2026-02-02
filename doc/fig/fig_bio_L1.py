import pandas as pd
import os.path
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter, defaultdict
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
from itertools import combinations, product
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec
from tueplots import bundles

# Styling
plt.rcParams.update(bundles.icml2024(column="full", nrows=2, ncols=2))

cluster_colors = {
    -1: "lightgrey",
     1: "orange",
     2: "#0069aa"
}

plot_color = "lightblue"

def within_cluster_dispersion(D, labels):
    D = np.asarray(D)
    labels = np.asarray(labels)
    W = 0.0
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        m = len(idx)
        if m <= 1:
            continue
        sub = D[np.ix_(idx, idx)]
        # mean of upper triangle (pairwise distances)
        tri = sub[np.triu_indices(m, k=1)]
        W += m * float(tri.mean())
    return W

# Data

df = pd.read_csv("../../dat/Bio_deep/sequences_metadata_cleaned.csv")
df_distances = pd.read_csv("../../dat/Bio_deep/aligned_dist.csv", index_col = 0).fillna(0)

# Processing of Distance matrix and metadata
df = df[df["Geo_Location"].notnull()]
counts = df["Geo_Location"].value_counts()
filtered = df[df["Geo_Location"].isin(counts[counts >= 15].index)]
labels = [lab for lab in df_distances.index if lab in df_distances.columns]
df_distances = df_distances.loc[labels, labels]
df_distances = df_distances.apply(pd.to_numeric, errors="coerce")
vals = df_distances.values.copy()
np.fill_diagonal(vals, 0.0)
i_lower = np.tril_indices_from(vals, k=-1)
vals[i_lower[1], i_lower[0]] = vals[i_lower]
vals = np.where(np.isnan(vals), vals.T, vals)
symm = pd.DataFrame(vals, index=df_distances.index, columns=df_distances.columns)


# Clustering

d = squareform(symm, checks=False)
Z = linkage(d, method="average")
labels = np.array(fcluster(Z, t=0.01, criterion="distance"))
counts = Counter(labels)


# Outlier removal
reject_label = -1
new_labels = np.array([
    lbl if counts[lbl] > 1 else reject_label
    for lbl in labels
])

# mapping
acc2label = dict(zip(symm.index, new_labels))
acc2country = {}
for _, row in filtered.iterrows():
    acc2country[row["Accession"]] = row["Geo_Location"]

country2labels = {}
for acc in acc2label:
    country = acc2country[acc]
    if(country not in country2labels.keys()):
        country2labels[country] = [acc2label[acc]]
    else:
        country2labels[country] += [acc2label[acc]]

rows = []
all_labels = set()

for country, labels in country2labels.items():
    c = Counter(labels)
    all_labels |= set(c.keys())
    rows.append({"country": country, "n": len(labels), "counts": dict(c)})

# MDS projection of clustering
df = pd.DataFrame(rows)
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=111,
    n_init=4,
    max_iter=300
)

XY = mds.fit_transform( np.asarray(symm))

#Pairwise ml distance

within = defaultdict(list)
between = defaultdict(list)
for i, j in combinations(symm.index, 2):
    d = symm.loc[i, j]
    if pd.isna(d): 
        continue
    c1, c2 = acc2country.get(i), acc2country.get(j)
    if c1 is None or c2 is None:
        continue
    if c1 == c2:
        within[c1].append(d)
    else:
        between[(c1,c2)].append(d)

within_dists = {}
for country in set(acc2country.values()):
    seqs = [acc for acc,c in acc2country.items() if c == country and acc in symm.index]
    if len(seqs) < 2:
        continue  # can't compute within-group distances for 1 seq
    vals = []
    for i,j in combinations(seqs, 2):
        vals.append(symm.loc[i,j])
    within_dists[country] = vals

wd_df = pd.DataFrame(
    [(c, v) for c, lst in within_dists.items() for v in lst],
    columns=["Country","ML Distance"])

# Log ODDs
country2fraction1_2 = {}
for _, row in df.iterrows():
    country = row["country"]
    counts = row["counts"]
    if(-1 not in counts.keys()):
        counts[-1] = 0
    if(2 not in counts.keys()):
        counts[2] = 0
    country2fraction1_2[country] = np.log((counts[2]+0.5) / (counts[1] + 0.5))

# Time cluster frequency
filtered.index = filtered["Accession"]
filtered["label"] = acc2label
filtered_label = filtered[~filtered["label"].isna()]
filtered_label["collection_year"] = pd.to_datetime(
    filtered_label["Release_Date"], errors="coerce"
).dt.year
df_time = filtered_label.dropna(subset=["collection_year", "label"])
df_time = df_time[df_time["label"] != -1]

freq = (df_time.groupby(["collection_year", "label"]).size().reset_index(name="n"))

years = range(freq["collection_year"].min(), freq["collection_year"].max() + 1)
labels = sorted(freq["label"].unique())

full_idx = pd.MultiIndex.from_product([years, labels],names=["collection_year", "label"])

freq_full = (freq.set_index(["collection_year", "label"]).reindex(full_idx, fill_value=0).reset_index())

counts = (freq_full.pivot(index="label", columns="collection_year", values="n").reindex(index=labels, columns=list(years), fill_value=0))

grouped = wd_df.groupby(by = "Country").mean()
grouped["logOdds"] = country2fraction1_2
grouped_filtered = grouped.dropna()

fig = plt.figure()
gs = GridSpec(
    nrows=3, ncols=2, figure=fig,
    width_ratios=[0.75, 1.25],  
    height_ratios=[0.75, 1.2, 0.1]
)

# Panel A (row 0, full width): boxplot
ax_box = fig.add_subplot(gs[0, 0:2])

# Panel B (row 1, col 0): MDS clusters + hulls
ax_mds = fig.add_subplot(gs[1, 0])

# Panel C (row 1, col 1): scatter + regression
ax_scatter = fig.add_subplot(gs[1, 1])

# Panel D (row 2, full width across left two cols): heatmap
ax_heat = fig.add_subplot(gs[2, 0:2])

# Plotting order boxplot
order = (
    wd_df.groupby("Country")["ML Distance"]
    .median()
    .sort_values()
    .index
)

# Boxplot
p1 = sns.boxplot(
    data=wd_df,
    x="Country",
    y="ML Distance",
    width=0.6,
    showcaps=True,
    boxprops={"facecolor": plot_color, "edgecolor": "black"},
    medianprops={"color": "black", "linewidth": 2},
    whiskerprops={"color": "black"},
    showfliers=False,
    order = order,
    ax = ax_box
)
p1.set(xlabel=None)

ax_box.tick_params(axis="x", rotation=0)
ax_box.set_title("Within-country pairwise sequence distances")
ax_box.grid(axis="y", linestyle="--", alpha=0.3)
ax_box.set_axisbelow(True)

# MDS projection
labels = np.unique(new_labels)

for lab in labels:
    pts = XY[new_labels == lab]
    color = cluster_colors.get(lab, "black")

    if lab == -1:
        ax_mds.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.6, color=color, label="Filtered")
        continue

    ax_mds.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.9, color=color, label=f"Cluster {lab}")

    if len(pts) >= 3:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        ax_mds.fill(
            hull_pts[:, 0], hull_pts[:, 1],
            alpha=0.1, edgecolor="black", color=color, linewidth=1
        )

ax_mds.set_xlabel("MDS 1")
ax_mds.set_ylabel("MDS 2")
ax_mds.set_title("MDS projection")

ax_mds.legend(frameon=True, loc="upper right")

# Scatterplot + regression
x = grouped["ML Distance"].values
y = grouped["logOdds"].values

sns.regplot(
    x=x, y=y, ax=ax_scatter,
    ci=95, n_boot=2000,
    scatter_kws=dict(s=40,color= plot_color ,edgecolor="black", alpha=0.85),
    line_kws=dict(linestyle="--", linewidth=2, color= plot_color)
)

texts = []

for country, row in grouped.iterrows():

    # hacky placement of labels by hand per country
    ha_ = "right"
    va_ = "center"
    offset = (-5, 0)
    if(country == "Latvia"):
        ha_ = "left"
        va_ = "top"
        offset = (5, 0)
    if(country == "Cambodia"):
        va_ = "bottom"
    if(country == "Japan"):
        va_ = "top"
    if(country == "Canada"):
        va_ = "bottom"
    if(country == "Mexico"):
        va_ = "top"
    if(country == "Nepal"):
        va_ = "top"
    if(country == "Guatemala"):
        va_ = "top"

    txt = ax_scatter.annotate(
        country,
        xy=(row["ML Distance"], row["logOdds"]),
        xytext=offset,
        textcoords="offset points",
        ha=ha_,
        va=va_,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        clip_on=False,
        fontsize=6
    )
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
    texts.append(txt)

ax_scatter.set_xlabel("Mean within-country sequence distance")
ax_scatter.set_ylabel("Log-odds")
ax_scatter.set_title("Diversity vs Log-odds")
ax_scatter.grid(True, linestyle="--", alpha=0.35)
ax_scatter.set_axisbelow(True)
ax_scatter.spines["top"].set_visible(False)
ax_scatter.spines["right"].set_visible(False)

#regression
coef = np.polyfit(x, y, 1)
r2 = r2_score(y, coef[0]*x + coef[1])
ax_scatter.text(
    0.03, 0.97, f"$R^2 = {r2:.2f}$",
    transform=ax_scatter.transAxes, ha="left", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9)
)

#heatmap
years = counts.columns
n_rows = len(counts)

M = counts.to_numpy().astype(float)
M = np.log1p(M)
base = np.array([mcolors.to_rgb(cluster_colors[lbl]) for lbl in counts.index]) 

denom = np.array([[M.max()]])
denom = np.where(denom == 0, 1, denom)
A = np.clip(M / denom, 0, 1)

white = np.ones((n_rows, M.shape[1], 3))
color = base[:, None, :]                     
rgb = white * (1 - A[..., None]) + color * A[..., None]

rgba = np.concatenate([rgb, np.ones((n_rows, M.shape[1], 1))], axis=2)

ax_heat.imshow(
    rgba,
    origin="upper",
    aspect="auto",
    interpolation="none",
    extent=(0, len(years), 0, n_rows)
)

ax_heat.set_xticks(np.arange(len(years) + 1), minor=True)
ax_heat.set_yticks(np.arange(n_rows + 1), minor=True)
ax_heat.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
ax_heat.tick_params(which="minor", bottom=False, left=False)

ax_heat.set_xticks(np.arange(len(years)) + 0.5)
ax_heat.set_xticklabels(list(years), rotation=0, ha="right")
ax_heat.set_yticks(np.arange(n_rows) + 0.5)
ax_heat.set_yticklabels([2,1])

ax_heat.set_xlim(0, len(years))
ax_heat.set_ylim(n_rows, 0)

ax_heat.set_xlabel("Collection year")
ax_heat.set_ylabel("Cluster")
ax_heat.set_title("Cluster sample counts over time")

for ax, letter in [(ax_box,"A"), (ax_mds,"B"), (ax_scatter,"C"), (ax_heat,"D")]:
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontweight="bold", va="bottom")

fig.savefig(
    "figure5_panel_overview.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.10
)

fig.savefig(
    "figure5_panel_overview.png",
    format="png",
    dpi=300,           
    bbox_inches="tight",
    pad_inches=0.10
)
