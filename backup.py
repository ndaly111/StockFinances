def chart_needs_update(chart_path, last_data_update):
    """
    Determines if a chart needs to be updated based on the last data update timestamp.

    :param chart_path: Path to the chart file.
    :param last_data_update: Timestamp of the last data update (can be a float, a datetime object, or NaN).
    :return: Boolean indicating whether the chart needs to be updated.
    """
    # Check if last_data_update is NaN (Not a Number)
    if pd.isna(last_data_update):
        print("last TTM update not available")
        # If the update timestamp is not available, assume the chart needs an update
        return True

    # Convert last_data_update to datetime if it's a float (Unix timestamp)
    if isinstance(last_data_update, float):
        last_data_update = datetime.fromtimestamp(last_data_update)


    # Convert last_data_update to datetime if it's a string
    elif isinstance(last_data_update, str):
        try:
            last_data_update = datetime.strptime(last_data_update, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If the date format is incorrect, return True to indicate the chart needs an update
            return True

    # If the chart file does not exist, it definitely needs an update
    if not os.path.exists(chart_path):
        print("missing file path")
        return True

    # Compare the last modification time of the chart with the last data update
    chart_mod_time = datetime.fromtimestamp(os.path.getmtime(chart_path))
    return true
        #(last_data_update > chart_mod_time)
