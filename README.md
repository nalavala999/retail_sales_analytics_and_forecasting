# Retail Sales Analytics & Forecasting

**Author:** NAGAMALLESWARA RAO ALAVALA  
**Date:** 11-02-2025

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Data Lineage](#data-lineage)
4. [Conceptual Mapping to Retail Tables](#conceptual-mapping-to-retail-tables)
5. [Gold Layer Models](#gold-layer-models)
6. [ETL Summary](#etl-summary)
7. [Power BI Deliverables](#power-bi-deliverables)
8. [Next Steps (Planned)](#next-steps-planned)
9. [Notes](#notes)

---

## Project Overview

This project implements an end-to-end **retail sales analytics** pipeline using the Superstore dataset (Orders, Customers, Products, Regions). It provides a clean **star schema** for BI, performance-oriented **pre-aggregations**, and foundations for **forecasting** and a **Gen-AI chatbot**.

**Tech stack**
- **Databricks**: Medallion architecture (Bronze, Silver)  
- **dbt**: Gold dimensional models (dims, facts, views, aggs)  
- **Power BI**: Dashboards (sales trend, profit heatmap, customer analysis)  
- **(Planned)** ML forecasting (next-month sales) & Gen-AI Q&A

**Goal**: Deliver trustworthy, fast, and explainable retail insights across **regions**, **products**, and **customers**.

---

## 🧩 Architecture & Data Flow

### 🔹 Bronze (Raw)
- **Schema:** `bronze`
- **Table:** `superstore` (raw CSV → Delta)
- **Purpose:** Immutable landing of the Kaggle Superstore dataset

### 🔹 Silver (Clean)
- **Schema:** `silver`
- **Table:** `superstore_clean`
- **Purpose:** Typed, trimmed, deduped order lines + derived fields (`order_ym`, `ship_days`)

### 🔹 Gold (Analytics)
- **Schema:** `gold`
- **dbt models:** Dimensions, fact table, views, and pre-aggregations for BI

---

## 🧭 Data Lineage

```mermaid
graph TD
    subgraph Bronze [Databricks - Bronze]
        b_superstore["bronze.superstore"]
    end

    subgraph Silver [Databricks - Silver]
        s_clean["silver.superstore_clean"]
    end

    subgraph Gold [dbt - Gold]
        dim_date["dim_date"]
        dim_customer["dim_customer"]
        dim_product["dim_product"]
        dim_region["dim_region"]
        dim_order["dim_order"]
        fact_sales["fact_sales"]
        agg_monthly_region_product["agg_monthly_region_product"]
        agg_customer_lifetime["agg_customer_lifetime"]
        agg_product_perf["agg_product_perf"]
        v_dim_*["v_dim_* (views)"]
        v_fact_sales["v_fact_sales (view)"]
    end

    %% Bronze → Silver
    b_superstore --> s_clean

    %% Silver → Dimensions
    s_clean --> dim_customer
    s_clean --> dim_product
    s_clean --> dim_region
    s_clean --> dim_order
    s_clean --> dim_date

    %% Silver → Fact
    s_clean --> fact_sales

    %% Dimensions → Fact
    dim_customer --> fact_sales
    dim_product --> fact_sales
    dim_region  --> fact_sales
    dim_order   --> fact_sales
    dim_date    --> fact_sales

    %% Facts → Aggregations / Views
    fact_sales --> agg_monthly_region_product
    fact_sales --> agg_customer_lifetime
    fact_sales --> agg_product_perf
    dim_* --> v_dim_*
    fact_sales --> v_fact_sales
