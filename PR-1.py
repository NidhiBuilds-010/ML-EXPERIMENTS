import statistics

# Given data
data = [10, 20, 20, 30, 40, 50, 20, 60, 70]

# Calculate statistical measures
mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)

# Display results
print("Given Data:", data)
print("Mean   =", mean)
print("Median =", median)
print("Mode   =", mode)