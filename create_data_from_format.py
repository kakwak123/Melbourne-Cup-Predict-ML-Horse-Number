"""
Script to create sample data from Melbourne Cup results format.

Handles data in the format:
Finish | No. | Horse | Trainer | Jockey | Margin | Bar. | Weight | Penalty | Starting Price
"""

import pandas as pd
import numpy as np
import json
import os
import sys

def parse_melbourne_cup_results(results_text: str, year: int = 2023) -> pd.DataFrame:
    """
    Parse Melbourne Cup results in the provided format.
    
    Args:
        results_text: Text containing race results
        year: Year of the race
        
    Returns:
        DataFrame with parsed data
    """
    lines = [line.strip() for line in results_text.strip().split('\n') if line.strip()]
    
    # Skip header line if present
    if 'Finish' in lines[0] or 'Finish' in lines[0].upper():
        lines = lines[1:]
    
    data = []
    for line in lines:
        parts = [p.strip() for p in line.split('\t')]
        
        if len(parts) < 10:
            continue
        
        finish = parts[0]
        horse_no = parts[1]
        horse_name = parts[2]
        trainer = parts[3]
        jockey = parts[4]
        margin = parts[5]
        barrier = parts[6]
        weight_str = parts[7]
        penalty = parts[8]
        price_str = parts[9]
        
        # Parse weight (remove 'kg')
        weight = float(weight_str.replace('kg', '').strip())
        
        # Parse starting price (remove '$')
        price = float(price_str.replace('$', '').strip())
        
        # Parse barrier
        barrier_num = int(barrier)
        
        # Estimate age (typically 4-8 for Melbourne Cup horses)
        age = np.random.randint(4, 9)
        
        # Generate performance stats (mock, but realistic)
        wins = np.random.randint(2, 15)
        places = np.random.randint(3, 20)
        
        # Generate recent race times (mock)
        recent_times = [120.0 + np.random.uniform(-2, 3) for _ in range(5)]
        
        # Track condition (most common is Good)
        track_conditions = ["Good", "Soft", "Heavy", "Firm"]
        track_condition = np.random.choice(track_conditions, p=[0.6, 0.2, 0.15, 0.05])
        
        data.append({
            'year': year,
            'horse_number': int(horse_no),
            'horse_name': horse_name.upper(),
            'age': age,
            'weight': weight,
            'trainer': trainer,
            'jockey': jockey,
            'barrier': barrier_num,
            'track_condition': track_condition,
            'distance': 3200,
            'wins': wins,
            'places': places,
            'recent_race_times': json.dumps(recent_times),
            'odds': price,
            'finishing_position': int(finish),
            'margin': margin if margin else '',
            'penalty': penalty
        })
    
    return pd.DataFrame(data)


def create_sample_from_format():
    """Create sample data files from the provided format."""
    
    # Sample data in the provided format
    sample_2023 = """Finish	No.	Horse	Trainer	Jockey	Margin	Bar.	Weight	Penalty	Starting Price
1	3	WITHOUT A FIGHT (IRE)	Anthony & Sam Freedman	Mark Zahra		16	56.5kg	1.0kg	$8
2	6	SOULCOMBE (GB)	Chris Waller	Joao Moreira	0.1L	15	53.5kg		$9
3	22	SERPENTINE (IRE)	Gai Waterhouse & Adrian Bott	Jye McNeil	0.5L	23	52.5kg		$15
4	1	VAUBAN (FR)	Willie Mullins	Ryan Moore	1.0L	3	58.0kg		$6.5
5	2	GOLD TRIP (FR)	Ciaron Maher & David Eustace	Ben Melham	1.2L	5	57.5kg		$7
6	12	ASHRUN (FR)	Andreas Wohler	Damien Oliver	1.5L	8	54.5kg		$12
7	8	BREAKUP (JPN)	Takeshi Matsushita	Tom Marquand	2.0L	12	55.0kg		$18
8	9	FUTURE HISTORY (IRE)	Ciaron Maher & David Eustace	Jamie Kah	2.5L	10	54.0kg		$20
9	11	MILITARY MISSION (IRE)	Gai Waterhouse & Adrian Bott	Damian Lane	3.0L	7	54.5kg		$22
10	15	RIGHT YOU ARE	Ciaron Maher & David Eustace	Michael Dee	3.5L	18	52.0kg		$25
11	14	VOW AND DECLARE	Danny O'Brien	James McDonald	4.0L	14	52.5kg		$30
12	20	ABSURDE (FR)	Willie Mullins	Zac Purton	4.5L	2	52.0kg		$35
13	7	YOUNG WERTHER (NZ)	Danny O'Brien	Blake Shinn	5.0L	9	55.5kg		$40
14	10	MAGICAL LAGOON (IRE)	Joseph O'Brien	Kerrin McEvoy	5.5L	11	54.0kg		$45
15	18	LASTOTCHKA (FR)	Mick Price & Michael Kent Jnr	Luke Currie	6.0L	20	52.0kg		$50
16	4	TRUE MARVEL (FR)	Anthony & Sam Freedman	Mark Zahra	6.5L	6	56.0kg		$55
17	13	OKITA SOUSHI (IRE)	Joseph O'Brien	Hugh Bowman	7.0L	13	54.0kg		$60
18	16	KALAPOUR (IRE)	Kris Lees	James McDonald	7.5L	19	52.0kg		$65
19	17	MORE FELONS (IRE)	Chris Waller	Joao Moreira	8.0L	21	52.0kg		$70
20	19	DAQIANSWEET JUNIOR (NZ)	Phillip Stokes	Ben Melham	8.5L	22	52.0kg		$75
21	21	POINT KING	Mick Price & Michael Kent Jnr	Damien Oliver	9.0L	4	52.0kg		$80
22	23	INTERPRETATION (IRE)	Ciaron Maher & David Eustace	Mark Zahra	9.5L	17	52.0kg		$85
23	24	JUST FINE (IRE)	Chris Waller	Jye McNeil	10.0L	1	52.0kg		$90
24	5	ALENQUER (FR)	Michael Moroney	Ryan Moore	10.5L	24	57.0kg		$95"""
    
    sample_2022 = """Finish	No.	Horse	Trainer	Jockey	Margin	Bar.	Weight	Penalty	Starting Price
1	21	GOLD TRIP (FR)	Ciaron Maher & David Eustace	Mark Zahra		14	57.5kg		$20
2	14	EMISSARY (GB)	Francis & William Musgrave	Patrick Moloney	0.2L	9	54.0kg		$26
3	3	HIGH EMOCEAN (NZ)	Ciaron Maher & David Eustace	Teo Nugent	0.4L	20	54.5kg		$31
4	12	DEAUVILLE LEGEND (IRE)	James Ferguson	Kerrin McEvoy	0.6L	18	57.5kg		$4.5
5	1	DUNADAM (IRE)	Joseph O'Brien	Jye McNeil	0.8L	11	58.0kg		$15
6	8	MONTEFILIA	David Payne	Jason Collett	1.0L	7	54.5kg		$18
7	5	STOCKMAN (NZ)	Joseph Pride	Sam Clipperton	1.2L	5	55.5kg		$25
8	11	VAUBAN (FR)	Willie Mullins	Ryan Moore	1.5L	16	58.0kg		$12
9	2	SMOKIN ROMANS (NZ)	Ciaron Maher & David Eustace	Jamie Kah	2.0L	13	57.5kg		$22
10	19	KNIGHTS ORDER (IRE)	Gai Waterhouse & Adrian Bott	Daniel Stackhouse	2.5L	3	54.0kg		$28
11	6	NUMERIAN (IRE)	Annabel Neasham	Tommy Berry	3.0L	12	55.5kg		$35
12	13	SERVELLO (IRE)	Joseph O'Brien	Ben Melham	3.5L	19	54.0kg		$40
13	10	GRAND PROMENADE (GB)	Ciaron Maher & David Eustace	Michael Dee	4.0L	15	54.5kg		$45
14	16	ARAPAHO (FR)	Bjorn Baker	Blake Shinn	4.5L	8	54.0kg		$50
15	9	LUNAR FLARE	Graeme Begg	Michael Poy	5.0L	10	54.5kg		$55
16	15	TRALEE ROSE (NZ)	Symon Wilde	Dean Yendall	5.5L	17	54.0kg		$60
17	20	SMOKEY DIAMONDS	Ciaron Maher & David Eustace	John Allen	6.0L	2	54.0kg		$65
18	4	HOO YA MAL (GB)	Gai Waterhouse & Adrian Bott	Blake Shinn	6.5L	6	56.0kg		$70
19	17	DAQWINS SWEET (NZ)	Phillip Stokes	Ben Melham	7.0L	4	54.0kg		$75
20	7	MONTEFILIA	David Payne	Jason Collett	7.5L	1	54.5kg		$80
21	18	LIGHTSABER	Peter Moody	Damien Oliver	8.0L	21	54.0kg		$85
22	22	CRYSTAL PEGASUS (GB)	Chris Waller	Joao Moreira	8.5L	22	52.0kg		$90
23	23	INTERPRETATION (IRE)	Ciaron Maher & David Eustace	Mark Zahra	9.0L	23	52.0kg		$95
24	24	MAKRAM (IRE)	Michael Moroney	Ryan Moore	9.5L	24	52.0kg		$100"""
    
    sample_2021 = """Finish	No.	Horse	Trainer	Jockey	Margin	Bar.	Weight	Penalty	Starting Price
1	4	VERRY ELLEEGANT (NZ)	Chris Waller	James McDonald		10	57.0kg		$18
2	2	INCENTIVISE	Peter Moody	Brett Prebble	4.0L	16	57.0kg		$3.5
3	22	SPANISH MISSION (USA)	Andrew Balding	Tom Marquand	4.2L	14	53.5kg		$12
4	3	FLOATING ARTIST (IRE)	Ciaron Maher & David Eustace	Fred Kersley	4.4L	11	56.5kg		$41
5	12	GRAND PROMENADE (GB)	Ciaron Maher & David Eustace	Kerrin McEvoy	4.6L	15	54.5kg		$31
6	1	DELPHI (IRE)	Anthony & Sam Freedman	Damien Oliver	4.8L	3	58.0kg		$14
7	10	THE CHOSEN ONE (NZ)	Murray Baker & Andrew Forsman	Daniel Stackhouse	5.0L	18	55.0kg		$26
8	13	TRALEE ROSE (NZ)	Symon Wilde	Dean Yendall	5.2L	12	54.5kg		$61
9	11	KNIGHTS ORDER (IRE)	Gai Waterhouse & Adrian Bott	Daniel Stackhouse	5.4L	7	54.5kg		$41
10	5	OCEAN BILLY (NZ)	Chris Waller	Damian Lane	5.6L	5	56.5kg		$51
11	6	TWILIGHT PAYMENT (IRE)	Joseph O'Brien	Jye McNeil	5.8L	20	58.0kg		$19
12	15	PONDUS (IRE)	Joseph O'Brien	Ben Melham	6.0L	19	54.0kg		$51
13	14	MASTER OF REALITY (IRE)	Joseph O'Brien	Michael Dee	6.2L	8	54.5kg		$81
14	20	THE GREATEST (AUS)	Ciaron Maher & David Eustace	Fred Kersley	6.4L	1	53.5kg		$91
15	16	MIAMI BOUND	Danny O'Brien	Michael Poy	6.6L	2	54.0kg		$101
16	8	PERFECT ALIBI	Michael Moroney	Jamie Kah	6.8L	17	55.5kg		$111
17	19	MIAMI BOUND	Danny O'Brien	Michael Poy	7.0L	13	54.0kg		$121
18	21	PORT GUILLAUME (FR)	Ben & JD Hayes	Blake Shinn	7.2L	9	53.5kg		$131
19	9	CHAPADA	Michael Moroney	Damien Oliver	7.4L	6	55.5kg		$141
20	7	VERRY ILLEGAL	Chris Waller	Kerrin McEvoy	7.6L	4	55.5kg		$151
21	17	SEA THE STARS	Ciaron Maher & David Eustace	Jamie Kah	7.8L	21	54.0kg		$161
22	18	THE CHOSEN ONE (NZ)	Murray Baker & Andrew Forsman	Daniel Stackhouse	8.0L	22	55.0kg		$171
23	23	MEDITERRANEAN (IRE)	Joseph O'Brien	Ben Melham	8.2L	23	52.0kg		$181
24	24	MONTEFILIA	David Payne	Jason Collett	8.4L	24	52.0kg		$191"""
    
    # Parse and save data
    os.makedirs("data/historical", exist_ok=True)
    
    for year, data_text in [(2023, sample_2023), (2022, sample_2022), (2021, sample_2021)]:
        print(f"\nProcessing {year} data...")
        df = parse_melbourne_cup_results(data_text, year)
        
        # Save to file
        output_path = f"data/historical/melbourne_cup_{year}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} horses to {output_path}")
        print(f"  Winner: {df[df['finishing_position'] == 1]['horse_name'].values[0]}")
    
    print("\n" + "="*60)
    print("Sample data files created successfully!")
    print("="*60)
    print("\nFiles created:")
    print("  - data/historical/melbourne_cup_2021.csv")
    print("  - data/historical/melbourne_cup_2022.csv")
    print("  - data/historical/melbourne_cup_2023.csv")
    print("\nYou can now train models with:")
    print("  python src/train_models.py")


if __name__ == "__main__":
    create_sample_from_format()

