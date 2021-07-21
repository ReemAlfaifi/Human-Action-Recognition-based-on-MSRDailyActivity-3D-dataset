#!/bin/sh
file_path="https://uowmailedu-my.sharepoint.com/personal/wanqing_uow_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwanqing%5Fuow%5Fedu%5Fau%2FDocuments%2FResearchDatasets%2FMSRDailyActivity3D&originalPath=aHR0cHM6Ly91b3dtYWlsZWR1LW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3dhbnFpbmdfdW93X2VkdV9hdS9FaXVIR1V6aHh1NUtuUzlId3lOQWd5TUJ6RUtFOUJuamZOZjVHVEdvbV94T293P3J0aW1lPXVUSmFhbHRNMlVn"
for i in {1..9}
do
  wget file_path+"MSRDailyAct3D_oack"+str(i)+".zip"
done