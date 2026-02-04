# FINAL-YEAR-RESEARCH-PROJECT

#Project Overview

Fashion apparel retailers operate across multiple price ranges to serve diverse customer segments and remain competitive in a highly dynamic market. However, inefficient pricing decisions often result in excessive inventory accumulation, frequent unplanned markdowns, and erosion of profit margins. These challenges are further intensified by rising manufacturing and sourcing costs driven by fluctuations in raw material prices, labour costs, logistics expenses, and supply-chain disruptions.

This project aims to address these issues by developing a data-driven pricing framework that identifies the optimal selling price for fashion apparel products to maximise revenue in the upcoming financial year, while accounting for historical sales behaviour, demand sensitivity, and cost pressures.

#Project Objectives

Analyse historical sales and pricing data to understand demand patterns

Identify price–demand relationships and price elasticity across product segments

Study seasonal trends, promotional effects, and product lifecycles

Reduce dependency on reactive markdown strategies

Develop a predictive pricing approach to support evidence-based decision-making

#Dataset Description

The dataset consists of transaction-level retail sales data stored in a single table, including:

Product attributes (Lifestyle, Colour, Size, Sleeve)

Transaction details (Date, Quantity, Selling Price, Discount)

Financial metrics (Gross Amount, Net Amount)

Unique product identifiers

Derived metrics such as revenue, discount percentage, price bands, and weeks since launch were created during preprocessing.

#Data Preprocessing

Data Quality Assessment

Missing value analysis

Duplicate and invalid record checks

Outlier detection for price, quantity, and revenue

#Exploratory Data Analysis (EDA)

The EDA phase focused on the following areas:

1. Univariate & Multivariate Analysis

Distribution analysis of prices, discounts, quantities, and revenue

Comparison across brands, product types, and price bands

2. Price–Demand Analysis

Price band creation using business-defined thresholds

Quantity-weighted average price and revenue calculations

Identification of demand variation across price levels

3. Time & Seasonality Analysis

Weekly and monthly sales trends

Impact of festive seasons, promotions, and end-of-season sales

4. Product Lifecycle Analysis

Sales lifecycle comparison between Fashion and Basic products

Identification of launch, peak, markdown, and clearance phases

#Key Insights

Fashion products exhibit a sharp post-launch demand spike followed by rapid decline, whereas basic products show steadier demand over time

Reactive discounting significantly impacts margins without guaranteeing proportional demand uplift

Quantity-weighted pricing metrics provide more reliable insights than simple averages

Seasonal effects and promotional periods strongly influence pricing effectiveness

#Tools & Technologies

Python (Pandas, NumPy, Matplotlib)

Jupyter Notebook

#Business Value

This project transforms pricing decisions from a reactive, intuition-driven process into a strategic, data-driven approach, enabling retailers to balance cost recovery, demand sensitivity, inventory management, and long-term revenue sustainability.
