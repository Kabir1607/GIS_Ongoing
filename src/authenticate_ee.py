import ee

if __name__ == '__main__':
    print("Starting Earth Engine Authentication...")
    ee.Authenticate()
    ee.Initialize(project='gis-hub-464402')

    # This will open a browser window for you to log into your Google Account.
    # Follow the instructions on-screen to grant Earth Engine access
    print("Authentication successful! You can now run the extraction script.")
