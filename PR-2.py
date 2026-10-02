import statistics

# Given data
data = [10, 20, 20, 30, 40, 50, 20, 60, 70]

# Calculate variance and standard deviation
variance = statistics.variance(data)
standard_deviation = statistics.stdev(data)

# Display results
print("Given Data:", data)
print("Variance =", variance)
print("Standard Deviation =", standard_deviation)