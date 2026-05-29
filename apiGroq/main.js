const API_KEY = '---';

let history = [];
let temp = 0.5;
function update_temperature_display(value) {
    document.getElementById('temp_number').textContent = `${value}`;
}
function change_temperature(input_temp) {
    temp = parseFloat(input_temp);
    update_temperature_display(temp);
    console.log(`Temperature updated to: ${temp}`);
}


function get_ai_response(user_input) {
    document.getElementById('input').value = '';
    if (user_input.trim() === 'clear') {
        history = [];
        document.getElementById('response').innerHTML = '';
        return;
    }
    const headers = {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json"
    };

    const data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers in Ukrainian. Be concise."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": temp
    };


    fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(data)
    })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error(response.statusText);
            }
        })
        .then(data => {

            const ai_response = data.choices[0].message.content;

            history.push({ role: 'user', content: user_input });
            history.push({ role: 'assistant', content: ai_response });
            document.getElementById('response').innerHTML = history.map(entry =>
                `<div class="${entry.role}_message">${entry.content}<br></div>`
            ).join('');
              
                
        }) 
        .catch (error => {

                document.getElementById('response').textContent = 'Error: ' + error.message;
            });
}