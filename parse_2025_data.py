"""
Parse 2025 Melbourne Cup lineup data and convert to standard format.
"""

import pandas as pd
import numpy as np
import json
import re

def parse_2025_lineup(text: str) -> pd.DataFrame:
    """Parse the 2025 Melbourne Cup lineup text."""
    
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    data = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for horse number pattern like "1. Al Riffa (19)"
        match = re.match(r'(\d+)\.\s+(.+?)\s+\((\d+)\)', line)
        if match:
            horse_num = int(match.group(1))
            horse_name = match.group(2).strip()
            barrier = int(match.group(3))
            
            # Get jockey and weight from next line
            jockey = ""
            weight = None
            trainer = ""
            odds = None
            
            if i + 1 < len(lines):
                jockey_line = lines[i + 1]
                # Extract jockey: "J: Mark Zahra 59kg"
                j_match = re.search(r'J:\s+(.+?)\s+(\d+\.?\d*)kg', jockey_line)
                if j_match:
                    jockey = j_match.group(1).strip()
                    weight = float(j_match.group(2))
            
            if i + 2 < len(lines):
                trainer_line = lines[i + 2]
                # Extract trainer: "T: Joseph O'Brien"
                t_match = re.search(r'T:\s+(.+)', trainer_line)
                if t_match:
                    trainer = t_match.group(1).strip()
            
            # Look for odds in subsequent lines
            if i + 3 < len(lines):
                odds_line = lines[i + 3]
                # Extract WIN odds (first number)
                odds_match = re.search(r'(\d+\.?\d*)', odds_line)
                if odds_match:
                    odds = float(odds_match.group(1))
            
            # Generate mock data for missing fields
            age = np.random.randint(4, 9)
            wins = np.random.randint(2, 15)
            places = np.random.randint(3, 20)
            recent_times = [120.0 + np.random.uniform(-2, 3) for _ in range(5)]
            track_condition = "Good"  # Default assumption
            
            data.append({
                'year': 2025,
                'horse_number': horse_num,
                'horse_name': horse_name.upper(),
                'age': age,
                'weight': weight if weight else 54.0,
                'trainer': trainer,
                'jockey': jockey,
                'barrier': barrier,
                'track_condition': track_condition,
                'distance': 3200,
                'wins': wins,
                'places': places,
                'recent_race_times': json.dumps(recent_times),
                'odds': odds
            })
            
            i += 4  # Skip processed lines
        else:
            i += 1
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 2025 Melbourne Cup data
    data_2025 = """Melbourne Cup 2025 Odds
Flemington R7
Open Race time:  3:00 pm, Today
1. Al Riffa (19)
J: Mark Zahra 59kg
T: Joseph O'Brien

8.00
WIN	
3.20
PLACE
2. Buckaroo (12)
J: Craig Williams 57kg
T: Chris Waller

9.00
WIN	
3.40
PLACE
3. Arapaho (15)
J: Rachel King 56.5kg
T: Bjorn Baker

41.00
WIN	
11.00
PLACE
4. Vauban (2)
J: Blake Shinn 56.5kg
T: Gai Waterhouse & Adrian Bott

21.00
WIN	
6.50
PLACE
5. Chevalier Rose (5)
J: Damian Lane 55.5kg
T: Hisashi Shimizu

51.00
WIN	
12.00
PLACE
6. Presage Nocturne (9)
J: Stephane Pasquier 55.5kg
T: Alessandro Botti

8.00
WIN	
3.20
PLACE
7. Middle Earth (13)
J: Ethan Brown 54.5kg
T: Ciaron Maher

34.00
WIN	
9.00
PLACE
8. Meydaan (22)
J: James McDonald 54kg
T: Simon & Ed Crisford

19.00
WIN	
6.00
PLACE
9. Absurde (4)
J: Kerrin McEvoy 53.5kg
T: Willie Mullins

21.00
WIN	
6.50
PLACE
10. Flatten The Curve (17)
J: Thore Hammer-Hansen 53.5kg
T: Henk Grewe

31.00
WIN	
8.50
PLACE
11. Land Legend (16)
J: Joao Moreira 53.5kg
T: Chris Waller

101.00
WIN	
21.00
PLACE
12. Smokin' Romans (11)
J: Ben Melham 53.5kg
T: Ciaron Maher

101.00
WIN	
21.00
PLACE
13. Changingoftheguard (24)
J: Tim Clark 53kg
T: Kris Lees

101.00
WIN	
21.00
PLACE
14. Half Yours (8)
J: Jamie Melham 53kg
T: Tony & Calvin McEvoy

6.50
WIN	
2.70
PLACE
15. More Felons (23)
J: Tommy Berry 53kg
T: Chris Waller

81.00
WIN	
17.00
PLACE
16. Onesmoothoperator (6)
J: Harry Coffey 53kg
T: Brian Ellison

19.00
WIN	
6.00
PLACE
17. Furthur (7)
J: Michael Dee 52kg
T: Andrew Balding

26.00
WIN	
7.50
PLACE
18. Parchment Party (3)
J: John R Velazquez 52kg
T: William I Mott

51.00
WIN	
12.00
PLACE
19. Athabascan (1)
J: Declan Bates 51.5kg
T: John O'Shea & Tom Charlton

101.00
WIN	
21.00
PLACE
20. Goodie Two Shoes (20)
J: W M Lordan 51.5kg
T: Joseph O'Brien

41.00
WIN	
11.00
PLACE
21. River Of Stars (14)
J: Beau Mertens 51.5kg
T: Chris Waller

15.00
WIN	
5.00
PLACE
22. Royal Supremacy (21)
J: Robbie Dolan 51kg
T: Ciaron Maher

34.00
WIN	
9.00
PLACE
23. Torranzino (18)
J: Celine Gaudray 51kg
T: Paul Preusker

34.00
WIN	
9.00
PLACE
24. Valiant King (10)
J: Jye McNeil 51kg
T: Chris Waller"""
    
    df = parse_2025_lineup(data_2025)
    
    # Save to processed directory
    import os
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/2025_lineup.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Parsed {len(df)} horses for 2025 Melbourne Cup")
    print(f"Saved to: {output_path}")
    print("\nFirst few entries:")
    print(df[['horse_number', 'horse_name', 'jockey', 'weight', 'trainer', 'barrier', 'odds']].head())

