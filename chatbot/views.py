from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import json
from .chatbot_logic import search_guidelines, classify_query, update_response

def chat_view(request):
    return render(request, 'chatbot/chat.html')

def process_message(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data.get('message')
        
        # Get chatbot response
        category = classify_query(user_input)
        results = search_guidelines(user_input)
        
        response = {
            'category': category,
            'results': results
        }
        
        return JsonResponse(response)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def feedback(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('query')
        response = data.get('response')
        feedback = data.get('feedback')
        
        update_response(query, response, feedback)
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
