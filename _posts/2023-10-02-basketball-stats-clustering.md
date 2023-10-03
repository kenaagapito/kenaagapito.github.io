---
title: Basketball Stats Clustering
date: 2023-10-02
categories: [Clustering,Unsupervised Learning]
tags: [clustering,sports,basketball,stats,unsupervised,outlier]
img_path: /assets/basketball-stats-clustering/
---
*This write-up is based on a project made by Ken Agapito, Joy Codia, and Almond Cruz as part of their coursework within the Master of Science in Data Science program of the Asian Institute of Management.*

In today's NBA, traditional playstyles are constantly evolving, leading to the emergence of new player positions and team compositions. While you're likely familiar with the classic five positions—center, power forward, small forward, shooting guard, and point guard—the game has evolved so much that it would be limiting to confine players strictly within these archetypes. Centers now extend their shooting range, guards are taking shots from even greater distances, and forwards are becoming more versatile, engaging in playmaking alongside their scoring duties. In short, player positions have become more fluid. I decided to find out if the data on player performance reflect this change. The task is to cluster players according to their basketball statistics, revealing distinct characteristics within each cluster. Such an analysis could benefit not only basketball team owners and coaches in scouting and building their rosters but also basketball enthusiasts who enjoy building teams in virtual games and community discussions.

## Data Source

The data came from [basketball-reference.com](https://basketball-reference.com) which contains the basketball statistics of players from the NBA and other prominent leagues. I focused on per-game average statistics of NBA players during the 2021-2022 regular season. The data can be downloaded as a CSV file through [this link](https://www.basketball-reference.com/leagues/NBA_2022_per_game.html) under the "Share & Export" button.

![figure-1](figure-1.png){: w="600" }
_Figure 1. Screenshot from basketball-reference.com._

## Data Pre-processing

Here are some pre-processing steps that I performed along with brief explanations:

+ I dropped the `team` and `age` features because they are unrelated to basketball statistics.
+ I removed the `position` feature because my objective is to reveal new archetypes and I did not want to rely on previous categorizations.
+ Players who played for multiple teams in the same season have different entries in the dataset: one for each team they played on and one for the aggregated statistics; I decided to include only the latter.
+ Some statistics are derived from other statistics. For example, `field goals` is the sum of `two-pointers` and `three-pointers`, and `total rebounds` is the sum of `offensive rebounds` and `defensive rebounds`. To avoid having multiple correlated features having a strong influence on the clustering algorithm, I decided to remove the derived statistics.

Here are the remaining features:

|Feature Name| Description| 
|:---:|:---:| 
|G|    Games 
|GS|    Games Started 
|MP|    Minutes Played 
|3P|    3-Point Field Goals 
|3PA|     3-Point Field Goal Attempts 
|2P|    2-Point Field Goals 
|2PA|    2-point Field Goal Attempts 
|FT|    Free Throws 
|FTA|    Free Throw Attempts 
|ORB|    Offensive Rebounds 
|DRB|    Defensive Rebounds 
|AST|    Assists 
|STL|    Steals 
|BLK|    Blocks 
|TOV|    Turnovers 
|PF|    Personal Fouls 
|PTS|    Points| 

## EDA

Figure 2 gives the histograms of the different basketball stats. Most positive skill stats (e.g. points, rebounds and assists) exhibit either a skewed bell curve or an exponential distribution. Both distributions imply the **presence of outliers**—players who extremely excel in certain stats. 

Features related to player usage are spread differently. `Games started` seem to follow a Pareto distribution. I infer that this is because **owners build teams around five elite players** who start in most games to maximize their skills. `Games played` show a skewed bimodal distribution, peaking at around 60-70 games and again at around 0-10 games. These two groups pertain to **regular players and substitutes** who only get to play towards the end of a game that is already decided (known as "garbage time").

![figure-2](figure-2.png)
_Figure 2. Histogram of basketball stats._

The correlation heat map in Figure 3 show some **notable strong positive correlations with obvious justifications**. For example, `shots attempted` and `shots made` have a strong positive correlation since a player is bound to score more if given more opportunities to score. `Minutes played` and `points` are also strongly correlated for a similar reason. Meanwhile, the **three-pointers has a strong negative correlation with offensive rebounds**. This is due to the nature of the game: players who frequently shoot three-pointers, being that they need to shoot far away from the basket, would less likely grab a rebound while playing offense.

![figure-3](figure-3.png){: w="450" }
_Figure 3. Correlation heat map of basketball stats._

## Dimensionality Reduction

I scaled the dataset to bring all the features to more or less the same magnitude. I chose a scaling technique called **robust scaling** which is less sensitive to outliers compared to typical methods such as min-max scaling and standard scaling. Then I transformed the features into **principal components**. Principal components are new dimensions that capture the variations of the data points.

Figure 4 shows a **scree plot** which gives the individual and cumulative explained variances of each principal component. The objective is to reduce the number of dimensions while still capturing as much variance as possible. The elbow of the individual explained variance occurs at around 3 principal components. At this number, the cumulative explained variance is already around 85%. This is a clear indication to **reduce the dimensions to the first 3 principal components**.

![figure-4](figure-4.png){: w="600" }
_Figure 4. Scree plot from principal component analysis._

To visualize the transformed data, the projection of the points and the features along the first 2 principal components is shown in Figure 5 as a **biplot**. Note that the third principal component contributes less than 10% explained variance, meaning the first 2 principal components, and consequently the biplot, already give an accurate description of the data.

![figure-5](figure-5.png){: w="500" }
_Figure 5. Biplot from principal component analysis._

The projections of the features in the biplot, i.e. the arrows, give us ample information about the first 2 principal components. Principal component 1 spans the horizontal component of the graph. The top 5 contributions come from the features with the largest horizontal components, and these are `free throws made`, `free throw attempts`, `two-pointers made`, `two-pointer attempts`, and `turnovers`. Features that contribute almost exclusively to principal component 1, i.e. features represented by almost horizontal arrows, are `free throw attempts`, `two-point attempts`, `games played`, and `games started`. These are strong indications that **principal component 1 highlights offensive productivity and player usage**.

Meanwhile, principal component 2 spans the vertical component of the graph and is influenced mostly by the features represented by almost vertical arrows: `offensive rebounds`, `blocks`, `three-pointer attempts`, `three-pointers made`, and `assists`. Note that offensive rebounds and blocks point almost opposite to three-pointers and assists. This implies that these sets of features are negatively correlated. This agrees with what was deduced from the correlation heatmap in Figure 3: players who play below the rim—also known as “big men”—do not have many opportunities to shoot three-pointers. Hence **principal component 2 separates big men (upper half of the graph) from scorers and playmakers (lower half of the graph)**.

## Clustering

### Representative-based clustering

The succeeding discussions focus on the results of different clustering algorithms applied to the data, starting with representative-based clustering, particularly **k-medoids** clustering. I chose k-medoids over the more standard k-means algorithm because the cluster center of the former will be an actual player instead of some point in the feature space. Figure 6 shows a visualization of the different cluster models. **All clustering results look well-separated**; this could be attributed to the strong contributions of the first 2 principal components to the variance of the data.

![figure-6](figure-6.png)
_Figure 6. Visualization of clusters as biplots._

I looked into **internal validation criteria** to determine the optimal number of clusters, shown in Figure 6. Without going into the details of what each criterion mean, the graph shows a dip in Davies-Bouldin (DB) index and a peak in silhouette coefficient and gap statistic when the number of clusters is equal to 5, which are strong indications that **the optimal numbner of clusters is 5**.

![figure-7](figure-7.png){: w="650" }
_Figure 7. Internal validation criteria versus number of clusters._

Figures 8, 9 and 10 show the optimal clustering with the biplot superimposed, the number of players per cluster, and the representative player (medoid) of each cluster. The medoid is an existing player that best describes its cluster on the average. Keeping in mind the meaning of the first 2 principal components, these figures help us identify key descriptions about each cluster.

![figure-8](figure-8.png){: w="700" }
_Figure 8. Visualization of optimal clustering as a biplot._

![figure-9](figure-9.png){: w="500" }
_Figure 9. Number of players in each cluster._

![figure-10](figure-10.png)
_Figure 10. Stats of representative player in each cluster._

**Cluster 1: Valuable players**

Together with bench warmers, valuable players are the **most represented demographic** in the population. They are mainly characterized by an average number of games played and games started. This could mean that they play whenever the starters need rest and start in the game whenever a starter cannot play. Some are expected to score and distribute the ball on the offensive end (lower half of the cluster) while some find their role under the rim (upper half of the cluster), but they are **not expected to produce the same numbers as star players**.

**Cluster 2: Big men**

Big men are characterized by high rebounds and blocks. They possess **excellent rim protection abilities**, using their size, athleticism, and instincts to grab a high number of boards and control the glass. Some of them also excel in scoring (right half of the cluster). They are dominant forces in the paint, often using their size and strength to bully their way through defenders. They are the **second smallest cluster in the data**, on account of the current playstyle in the NBA that prioritizes spreading players on the floor and executing fluid offense over clogging the paint.

**Cluster 3: Superstars**

Superstars play and start in most games, play major minutes per game, and produce exceptionally on offense. They are the **smallest cluster in the data**, owing to the fact that they are considered **outliers in the numbers that they produce**. Being highly skilled and recognized forces on the court, these players can score and/or create scoring opportunities for their teammates. Some of them possess excellent ball-handling skills and the ability to drive to the basket, some are deadly spot-up three-point shooters, and some can be a mix of both. Superstars are the greatest assets of a team and a major factor in a team's success.

**Cluster 4: Bench warmers**

Together with valuable players, bench warmers are the most represented demographic in the population. Bench warmers **do not start games nor produce the numbers to make a significant impact** on the team. These players often find themselves in a difficult position as they may not receive regular playing time but are still expected to perform when called upon. These players need to put in extra effort in practice, stay focused, and take advantage of any opportunities that arise to demonstrate their skills and impact the game.

**Cluster 5: Star scorers**

Star scorers **play similar minutes to superstars but with slightly less scoring output**. They include starting guards and sixth men. They are well-rounded players whose role is to score and involve their teammates. These players spread the floor on the offensive end and are the go-to players of a team whenever their superstar is struggling.

## Conclusion

While the clustering did not give us a definitive set of new basketball positions, the principal components did point us to the specific set of statistics that differentiate players. The first principal component relates to player usage and offensive productivity, while the second principal components distinguishes big men from scorers and playmakers, influencing the clusters accordingly. This does not mean that we failed in our task; rather, I see this as further proof of the fluidity of basketball positions. In the traditional positions, we'd expect clusters based on rebounds, inside scoring, outside scoring, and assists. Yet the clustering results were only able to isolate the rebounders. This reaffirms the versatile nature of modern basketball roles, with all players (except big men, who focus on rim protection) capable of scoring, facilitating plays, and securing rebounds. I hope this analysis gave you a richer perspective on the ever-evolving game of basketball.