import os
import boto3
import json
import sys

client = boto3.client('rekognition')

summary = []
if len(sys.argv) <  2:
    print("No jobID or number of results given. Need JobId and number of results")
    quit()
else:
    print("JobID: ", sys.argv[1])
    print("MaxResults", sys.argv[2])
    response = client.get_face_search(
    JobId=sys.argv[1],
    MaxResults=int(sys.argv[2])
    )
    responsej = json.dumps(response, indent=4, sort_keys =True)            
    f = open('job-faces-output', 'w')
    f.write(responsej)
    f.close 
    #Here check if job still in progress, if so, print the output and getout.
    if  not response['Persons']:
            print(responsej)
            quit()
        #If here, job is completed.
    long=len(response['Persons'])
    fx = open('faceoutput.csv','w')
    fx.close()
        
    for res in range(0, long):
            b = response['Persons'][res].get('FaceMatches','')
            bb = response['Persons'][res].get('Person','')
            c = json.dumps(b)
            if len(b) > 0:
                summary.append(b[0]['Face']['ExternalImageId'])
                
                #extract coordenates to export to numpy and OpenCSV
                cx=bb['Face']['BoundingBox']['Left']
                cy=bb['Face']['BoundingBox']['Top']
                cx2=bb['Face']['BoundingBox']['Width']
                cy2=bb['Face']['BoundingBox']['Height']

                #extract confidence value 
                conf=b[0]['Similarity']
                
                #logging to CSV for OpenCSV later analysis, loading into numpy
                fx = open('facematch-output.csv','a')
                linetowrite = str(response['Persons'][res].get('Timestamp',"No time")) + "," + str(cx) + "," + str(cy) + "," + str(cx2) + "," + str(cy2) + "," + str(conf) + "\n"
                fx.write(linetowrite)
            
    fx.close            
    summaryt = [ (summary.count(x), x) for x in set(summary)]

#Automatically upload the csv file to my S3 bucket.
data=open('facematch-output.csv','rb')
s3 = boto3.resource('s3')
s3.Bucket('yourbucketname’).put_object(Key='nameyouchoose.csv', Body=data)
