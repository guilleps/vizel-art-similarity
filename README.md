# Artshift

### Origen de datos de imágenes

#### 1. **Nombre del dataset o fuente:** *AI-ArtBench Detection and Attribution of AI-generated A*  
- **Enlace:** [https://www.kaggle.com/datasets/ravidussilva/real-ai-art](https://www.kaggle.com/datasets/ravidussilva/real-ai-art)  
- **Cantidad de imágenes:** 60,000  
- **Tipo de contenido:** Pintura de estilo art-nouveau, baroque, expressionism, impressionism, post-impressionism, realism, renaissance, romanticism, surrealism, ukiyo-e  
- **Notas:** Imágenes con resolución de 256x256, (datos de su trata...)

## Overview
This project showcases an experimental pipeline for analyzing and comparing Impressionist-style paintings using low-level visual features and embedding-based representations.

## Features
- **Compositional similarity**: Measures structural and compositional resemblances between artworks.
- **Low-level visual features**: Extracted from color transformations (color heatmap, hue, saturation, brightness) and texture descriptors (contrast, texture) to emphasize salient characteristics.
- **Embeddings**: Dense vector representations used for efficient similarity search, clustering, and retrieval.

## Workflow

![Big Picture App Web](./resources/bigpicture_v2.png)

The application's workflow includes:
1. **Apply transformations**: Preprocess images and compute visual transforms.
2. **Feature extraction**: Numerically extract visual and contextual features using deep neural networks.
3. **Similarity scoring**: Compute similarity scores and rankings between artworks.

## Directory Structure
```
└── 📁.
    └── 📁backend                # application server: endpoints, business logic, and configuration
        └── 📁api               # backend API handlers
        └── 📁backend           # backend core modules
        └── 📁tests             # unit and integration tests
    └── 📁frontend
        └── 📁src
            └── 📁components
            └── 📁pages
            └── 📁services
    └── 📁resources             # static assets (images, diagrams, docs)
    └── 📁services
        └── 📁service-transform # image transformation service
        └── 📁service-cnn       # feature-extraction service using neural networks
    └── 📁similarity_processor
        └── 📁api
        └── 📁similarity_processor
```

## Data security diagram

![Big Picture App Web](./resources/graphic_v2.png)