## Node Schema

| Field Name            | Description |
|----------------------|-------------|
| `user_id`             | Unique user identifier |
| `username`            | Discord username |
| `avg_text_embedding`  | Mean embedding vector of all user messages |
| `activity_stats`      | Aggregated activity features: total message count, interaction reply count, interaction initiation count |
| `role_embedding`      | Learned user embedding produced by the GNN (not stored) |

---

## Edge Schema

| Field Name        | Description |
|------------------|-------------|
| `edge_id`         | Unique edge identifier |
| `source_user_id`  | User initiating the interaction (replier or mentioner) |
| `target_user_id`  | User being replied to or mentioned |
| `timestamp`       | Time of the interaction |
| `edge_type`       | Interaction type: `mention` or `temporal` |
| `metadata`        | (Optional) Additional edge metadata (e.g. `has_media`) |
| `edge_confidence` | (Optional) Confidence weight (e.g. `1.0` for mention, `0.7` for temporal) |

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