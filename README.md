## Project Summary

This project explores social structure analysis using Graph Neural Networks applied to Discord interaction data.

User messages are modeled as a graph where:
- nodes represent users
- edges represent interactions (derived as temporal (time-based) replies and mentions)

Each user node is initialized with semantic features derived from message text (embedded using sentence transformers) and activity statistics (computed). A GraphSAGE-based GNN is then trained using self-supervised contrastive learning (observed is good, randomly sampled is bad) to produce embeddings that encode interaction structure and user roles.

The resulting embeddings are analyzed to uncover:
- social centrality
- bridging behavior between groups
- general structure and roles within the community

The project focuses on representation learning and structural analysis rather than meeting a supervised objective or task.

**Read [`graph.md`](./graph.md) for specific schemas and details.**

### WIP
- Other metrics (centrality and bridge score done)

