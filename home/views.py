from django.shortcuts import render
import pickle
import numpy as np


def home(request):
    if request.method=='POST':
        vintage = request.POST['vintage']
        transaction = request.POST['transaction']
        current_credit = request.POST['current credit']
        previous_credit = request.POST['previous credit']
        current_debit = request.POST['current debit']
        previous_debit = request.POST['previous debit']
        current_balance = request.POST['current balance']
        age = request.POST['age']
        occupation = request.POST['occupation']
        net_worth = request.POST['worth']

        scaler = pickle.load(open('static/minmax.pickle', 'rb'))

        a=scaler.transform(np.array([vintage,transaction,current_credit,previous_credit,current_debit,previous_debit,current_balance,age,occupation,net_worth]).reshape(-1,1).T)
        model=pickle.load(open('static/random_forest_regression_model.pkl', 'rb'))
        val = model.predict(a)
        if val ==1:

            context = {
            'message':"Ohh!! Average balance of customer falls below minimum balance in the next quarter",
        }
        else:
            context = {
            'message':"yeee Average balance of customer not falls below minimum balance in the next quarter",
        }



        return(render(request,'home.html',context))
    
    return(render(request,'home.html'))


 