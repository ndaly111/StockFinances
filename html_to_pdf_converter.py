#end of html_to_pdf_converter.py

import pdfkit
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def html_to_pdf(html_file_path, output_pdf_path):
    print("html to pdf converter 1 html to pdf")
    # Configuration for pdfkit
    # If wkhtmltopdf is in your system's PATH, or you have it installed in the default location, you might not need this.
    # Otherwise, provide the full path to wkhtmltopdf executable.
    config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')

    options = {
        'page-size': 'A4',
        'dpi': 96,
        'no-outline': None,
        'enable-local-file-access': None,
        'enable-javascript': None,
        'javascript-delay': 1000,
        'no-stop-slow-scripts': None,
        'enable-external-links': None,
        'enable-internal-links': None,
    }
    try:
        pdfkit.from_file(html_file_path, output_pdf_path, options=options, configuration=config)
        print(f"PDF successfully created at {output_pdf_path}")
    except Exception as e:
        print(f"An error occurred during PDF creation: {e}")
        logging.error(f"PDF creation error: {e}")


#end of html_to_pdf_converter.py