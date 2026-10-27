# Data Dictionary / Interpretation Guide

The exact columns can be inspected with:
python -c "import pandas as pd; print(pd.read_csv('dataset/train.csv').columns.tolist())"

Typical fields in the airline satisfaction dataset include:
- Gender: passenger gender
- Customer Type: loyal/disloyal customer category
- Age: passenger age
- Type of Travel: business/personal
- Class: travel class
- Flight Distance: distance of flight
- Inflight wifi service: rating
- Departure/Arrival time convenient: rating
- Ease of Online booking: rating
- Gate location: rating
- Food and drink: rating
- Online boarding: rating
- Seat comfort: rating
- Inflight entertainment: rating
- On-board service: rating
- Leg room service: rating
- Baggage handling: rating
- Checkin service: rating
- Inflight service: rating
- Cleanliness: rating
- Departure Delay in Minutes: delay
- Arrival Delay in Minutes: delay
- satisfaction: target
