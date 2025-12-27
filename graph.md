## Node schema

user_id               (username)  
avg_text_embedding    (mean vector of all user messages)
activity_stats        (total message count, interaction reply count, interaction initiation count)

Eventually we can get

role_embedding        (learned output, NOT stored)

## Edge schema

edge_id
source_user_id   (replier)
target_user_id   (previous speaker / mentioned user)
timestamp
edge_type        (mention | temporal)
metadata         (time_delta, is_media)
<!-- maybe add this later: edge_confidence (1.0 mention, 0.7 temporal) -->

## Trends to look for that can be represented, analyzed, modeled using a GNN
- Social structure and roles
- Interaction dynamics
- Community formation
- Behavioural consistency

## GNN Models
`z_u` = learned embedding representing user `u`.
It encodes the position of the user in the interaction graph.
conditioned on:
- who they interact with
- how interaction structure propagates
- initial node features (text + stats)

## Embedding Metrics (User-Level)
### Similarity / Affinity
Measures structural similarity between two users.
- cosine(z_u, z_v)

### Directional Influence (Asymmetry)
Measures dominance / adaptation.
- influence(u → v) = z_u · z_v
- asymmetry(u, v) = (z_u · z_v) − (z_v · z_u)

### Centrality (Embedding Space)
Graph-aware importance score.
- centrality(u) = mean cosine(z_u, z_all_users)

### Neighborhood Consistency
How coherent a user’s interaction neighborhood is.
- neighborhood_variance(u) = variance({z_v | v ∈ neighbors(u)})

### Role Clustering
Unsupervised role discovery.
- Input: [z_u || activity_stats_u]
- Algorithms: k-means, HDBSCAN

### Bridge Score
Identifies connectors between groups.
- bridge_score(u) = similarity(z_u, multiple cluster centroids)

### Interaction Preference
Affinity likelihood between users.
- P(u → v) ∝ z_u · z_v

### Temporal Change (if time-sliced)
Measures role / behavior shifts.
- Δ(u) = ||z_u(t+1) − z_u(t)||

### Notes
- Do **not** interpret individual embedding dimensions.
- All meaning comes from distances, dot products, and geometry **between** individual user embeddings.