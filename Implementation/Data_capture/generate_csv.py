import csv
import os

def get_output_csv(data):

    #remove /app/ to get output in local machine without using containers
    os.makedirs("/app/output", exist_ok=True)

    output_file = "/app/output/network_stats.csv"
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["mac_fill", "flood_pressure", "avg_age"])

        # Append rows
        writer.writerows(data)

    print("Data appended successfully!")
