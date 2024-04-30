library(rpart)
library(rpart.plot)
library(caret)
library(rsample)
library(dplyr)
library(ggplot2)

#The objective of this project is to predict prices for diamonds based on the color, carat, cut, and clarity.

file <- read.csv("/Users/nikhil/Downloads/diamonds.csv")

# Filter out unused columns
file <- file[-c(1, 6, 7, 9, 10, 11)]

set.seed(12345)

# Filter carats to be from 0 to 2.0
filtered_data <- file %>%
  filter(carat >= 0 & carat <= 2.0)

# Contingency Table for Color vs. Price
color_price <- aggregate(price ~ color, data = filtered_data, FUN = mean)
color_price

# Contingency Table for Color vs. Carat
color_carat <- aggregate(carat ~ color, data = filtered_data, FUN = mean)
color_carat

# Contingency Table for Cut vs. Price
cut_price <- aggregate(price ~ cut, data = filtered_data, FUN = mean)
cut_price

# Contingency Table for Cut vs. Carat
cut_carat <- aggregate(carat ~ cut, data = filtered_data, FUN = mean)
cut_carat

# Contingency Table for Clarity vs. Price
clarity_price <- aggregate(price ~ clarity, data = filtered_data, FUN = mean)
clarity_price

# Contingency Table for Clarity vs. Carat
clarity_carat <- aggregate(carat ~ clarity, data = filtered_data, FUN = mean)
clarity_carat

# Visual Comparison for Color vs. Price
ggplot(color_price, aes(x = color, y = price)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Average Price by Color",
       x = "Color",
       y = "Average Price") +
  theme_minimal()

# Visual Comparison for Color vs. Carat
ggplot(color_carat, aes(x = color, y = carat)) +
  geom_bar(stat = "identity", fill = "lightgreen", width = 0.5) +
  labs(title = "Average Carat by Color",
       x = "Color",
       y = "Average Carat") +
  theme_minimal()

# Visual Comparison for Cut vs. Price
ggplot(cut_price, aes(x = cut, y = price)) +
  geom_bar(stat = "identity", fill = "salmon", width = 0.5) +
  labs(title = "Average Price by Cut",
       x = "Cut",
       y = "Average Price") +
  theme_minimal()

# Visual Comparison for Cut vs. Carat
ggplot(cut_carat, aes(x = cut, y = carat)) +
  geom_bar(stat = "identity", fill = "gold", width = 0.5) +
  labs(title = "Average Carat by Cut",
       x = "Cut",
       y = "Average Carat") +
  theme_minimal()

# Visual Comparison for Clarity vs. Price
ggplot(clarity_price, aes(x = clarity, y = price)) +
  geom_bar(stat = "identity", fill = "lightcoral", width = 0.5) +
  labs(title = "Average Price by Clarity",
       x = "Clarity",
       y = "Average Price") +
  theme_minimal()

# Visual Comparison for Clarity vs. Carat
ggplot(clarity_carat, aes(x = clarity, y = carat)) +
  geom_bar(stat = "identity", fill = "lightblue", width = 0.5) +
  labs(title = "Average Carat by Clarity",
       x = "Clarity",
       y = "Average Carat") +
  theme_minimal()

# Create the scatterplot with filtered data
ggplot(filtered_data, aes(x = carat, y = price, color = color)) +
  geom_point(alpha = 0.5) +  # Scatterplot points with transparency
  labs(title = "Price vs Carat by Color (0-3 Carat Range)", x = "Carat", y = "Price", color = "Color") +
  theme_minimal()

# Scatterplot for price vs carat by clarity
ggplot(filtered_data, aes(x = carat, y = price, color = clarity)) +
  geom_point(alpha = 0.5) +
  labs(title = "Price vs Carat by Clarity", x = "Carat", y = "Price", color = "Clarity") +
  theme_minimal()

# Scatterplot for price vs carat by cut
ggplot(filtered_data, aes(x = carat, y = price, color = cut)) +
  geom_point(alpha = 0.5) +
  labs(title = "Price vs Carat by Cut", x = "Carat", y = "Price", color = "Cut") +
  theme_minimal()
#create model
model <- rpart(price ~ color + cut + clarity + carat, data = filtered_data)
rpart.plot(model)

# 5 fold cv
train_control <- trainControl(method = "cv", number = 5)
model_caret <- train(
  price ~ color + cut + clarity + carat, 
  data = filtered_data, 
  method = "rpart",  # Use rpart method
  trControl = train_control
)

#80% of the values will be training set.
train_size <- round(0.8 * nrow(filtered_data))
train_indices <- sample(seq_len(nrow(filtered_data)), size = train_size)
train_data <- filtered_data[train_indices, ]
test_data <- filtered_data[-train_indices, ]

#Save predictions and generate accuracy
predictions <- predict(model_caret, newdata = test_data)
accuracy <- postResample(predictions, test_data$price)
print(accuracy)
