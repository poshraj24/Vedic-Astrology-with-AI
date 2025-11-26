flowchart LR

    %% Inputs
    A[Palm Picture] --> B[Feature Extraction]
    B --> C[Palm Features<br/>(Feature 1..N)]

    D[Birth Details] --> E[Kundali Generation<br/>(Features 1..N)]
    E --> F[Numerology]

    %% Offline / Knowledge base block
    subgraph K[Offline Knowledge & Scriptures]
        G[(KB for Palmistry)]
        H[(BSP)]
        I[(Śāstras)]
    end

    %% Connections into KB
    C --> G
    E --> H
    F --> I

    %% Output model
    G --> J[Predictive Model]
    H --> J
    I --> J
