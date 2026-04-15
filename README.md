# Customer Segmentation System

A machine learning web application that segments customers into behavioural clusters using KMeans clustering, deployed via a FastAPI dashboard.

---

## Project Structure

```
Case Study/
├── Deployment/
│   ├── templates/
│   │   └── index.html          # Jinja2 HTML dashboard
│   ├── main.py                 # FastAPI application
│   ├── app.py                  # Original Streamlit prototype
│   ├── requirements.txt        # Python dependencies
│   ├── kmeans_model.pkl        # Trained KMeans model (3 clusters)
│   ├── scaler.pkl              # StandardScaler
│   ├── pca_model.pkl           # PCA (18 components)
│   ├── features.pkl            # Ordered list of 30 feature columns
│   ├── customer_ids.pkl        # Customer ID Series (C1 – C10000)
│   └── training_data.pkl       # Feature matrix (10,000 × 30)
├── case study.ipynb            # Model training notebook
├── segmentation_model_data.xlsx
└── segmentation_model_data_description.xlsx
```

---

## ML Pipeline

```
Raw Customer Data
      │
      ▼
StandardScaler  ──►  PCA (18 components)  ──►  KMeans (k=3)  ──►  Cluster Label
```

| Step | Tool | Detail |
|---|---|---|
| Feature scaling | `StandardScaler` | Normalises all 30 features to zero mean, unit variance |
| Dimensionality reduction | `PCA` | Reduces 30 features to 18 principal components |
| Clustering | `KMeans` | 3 clusters fitted on PCA-transformed data |

---

## Segments

| Cluster | Segment | Description |
|---|---|---|
| 0 | Dormant / Low Engagement Users | Minimal activity across digital channels. Candidates for re-engagement campaigns. |
| 1 | High-Value Digital Power Users | High transaction volumes, strong digital product adoption. Prioritise retention and premium offers. |
| 2 | Active Mass Market Users | Regularly active on core services. Strong candidates for upselling and feature adoption. |

---

## Features (30)

| Feature | Description |
|---|---|
| `age_in_loop` | Customer tenure on platform |
| `usr_age` | User age (log-transformed) |
| `bill_count` | Number of bill payments |
| `has_virtual_card` | Virtual card ownership flag |
| `has_card_txn` | Card transaction activity flag |
| `has_bill` | Bill payment activity flag |
| `net_flow_3m` | Net cash flow over 3 months |
| `net_flow_6m` | Net cash flow over 6 months |
| `flow_consistency` | Consistency of cash flow |
| `inflow_outflow_ratio_3m` | Ratio of inflows to outflows (3m) |
| `txn_activity_3m` | Transaction activity score (3m) |
| `avg_txn_value_3m` | Average transaction value (3m) |
| `spend_per_txn_3m` | Spend per transaction (3m) |
| `inflow_growth` | Growth rate of inflows |
| `outflow_growth` | Growth rate of outflows |
| `lipa_ratio_3m` | Lipa na M-PESA usage ratio (3m) |
| `card_usage_ratio` | Card usage frequency ratio |
| `biz_ratio_3m` | Business transaction ratio (3m) |
| `product_engagement` | Breadth of product usage |
| `investment_ratio` | Investment product usage ratio |
| `goals_ratio` | Savings goals activity ratio |
| `credit_utilization` | Credit facility utilisation rate |
| `high_spender_flag` | Binary flag for high spenders |
| `is_low_activity` | Binary flag for low-activity customers |
| `is_power_user` | Binary flag for power users |
| `flow_volatility` | Volatility of cash flow |
| `service_diversity` | Number of distinct services used |
| `digital_maturity` | Digital channel adoption score |
| `spending_pressure` | Ratio of outflows to total flow |
| `high_value_customer` | Binary flag for high-value customers |

---

## Setup & Run

### 1. Install dependencies

```bash
cd "Case Study/Deployment"
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn main:app --reload
```

### 3. Open the dashboard

```
http://127.0.0.1:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the dashboard with customer dropdown |
| `POST` | `/predict` | Accepts `customer_number` form field, returns segment result and customer snapshot |

---

## Dashboard Features

- Dropdown of all 10,000 customer IDs
- One-click segment prediction
- Colour-coded cluster badge (red / green / blue)
- Segment name and contextual business insight
- Full 30-feature customer snapshot table

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `jinja2` | HTML templating |
| `python-multipart` | Form data parsing |
| `scikit-learn` | KMeans, StandardScaler, PCA |
| `joblib` | Model serialisation |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
