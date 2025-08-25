def get_text_from_pdf :
  pdf_path_1 = "/content/RJIO_22_23.pdf"
  pdf_path_2 = "/content/RJIO_23_24.pdf"
  
  def extract_text_from_pdf(pdf_path):
      text = ''
      with pdfplumber.open(pdf_path) as pdf:
          for page in pdf.pages:
              text += page.extract_text() + '\n'
      return text
  
  pdf1_text = extract_text_from_pdf(pdf_path_1)
  pdf2_text = extract_text_from_pdf(pdf_path_2)
  print(pdf1_text[:1000])
  return pdf1_text+pdf2_text
