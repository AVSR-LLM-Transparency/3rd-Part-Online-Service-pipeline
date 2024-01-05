# ReadME
This is the contrast pipeline duirng the user study, calling Google Cloud Speech recognition Service and OpenAI ChatGPT-3.5 turbo API.     

It's super easy to run this pipeline as it just contains a single python script.               
But before you officially start, please make sure that you have the Google Cloud Service key and OpenAI key activated. If not, please google the steps to do it.       
After that, you need to configure 2 things:              
1. configure the GOOGLE_APPLICATION_CREDENTIALS environmental variable in the _.bashrc_ file.                        
2. change the _openai.api_key_ variable at line 214 in the code.              

Now, you can run the code directly in the terminal!                    
`python3 pipeline.py`                  

The user data will be saved in the _data_ folder, as included in the repo.                           

Good Luck!        
