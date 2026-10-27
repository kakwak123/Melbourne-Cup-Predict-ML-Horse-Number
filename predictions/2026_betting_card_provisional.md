# Melbourne Cup 2026: $100 card (provisional, prices as of 27 Oct)

Race: Tuesday 3 November 2026, Flemington, 3:00pm AEDT. Barrier draw: Saturday 31 October.
Prices: Ladbrokes all-in win odds from the likely-field page; they will move.

## What the evidence supports (and doesn't)

- 6 Cups of Betfair prices (2020-2025), verified results, leave-one-year-out scoring.
- Nothing beat guessing: not the market, not weight/barrier, not a blend. There is no edge to exploit.
- The favourite lost all six Cups. Winners were the market's 4th, 4th, 7th, 9th, 11th and 23rd choice. 13 of 18 placegetters came from outside the top three.
- So the plan is built around that shape (spread across the middle and long end), not around picking a winner.
- The bet shapes were chosen after seeing these same six Cups, so every number below is in-sample, not a forecast.
- Expected result with bookmaker margin: a loss of roughly $10-$45 of the $100. Treat it as a fun budget.

## Card A, the swing: ten $10 win bets (recommended)

Backtest, $100 per Cup: returned $299 (2020), $246 (2022), $0 in the other four. ROI -9% at exchange prices, -20% at bookmaker prices. Profitable in 2 of 6 Cups.

| Market rank | Horse                         | Price | Pays if it wins ($10) |
| ----------- | ----------------------------- | ----- | --------------------- |
| 8           | Enfjaar (IRE)                 | $17   | $170                  |
| 9           | Hopewell Rock (IRE)           | $17   | $170                  |
| 10          | Stinger Glass (JPN)           | $17   | $170                  |
| 11          | Middle Earth (GB)             | $21   | $210                  |
| 12          | Goodie Two Shoes (IRE)        | $26   | $260                  |
| 13          | Mr Monaco (GB)                | $26   | $260                  |
| 14          | Sons And Lovers (GB)          | $26   | $260                  |
| 15          | Zahrann (IRE)                 | $26   | $260                  |
| 16          | Zakouma                       | $34   | $340                  |
| 17          | Knight's Choice (2024 winner) | $51   | $510                  |

Rank ties are arbitrary (several horses share a price). The rule is: take the market's 8th to 17th shortest prices and put $10 on each to win.

## Card B, pays more often, loses more: spread + place + trifecta lottery

$60 win ($6 on ranks 8-17), $24 place ($8 each on ranks 4-6: James J Braddock, Saint George, Shockletz), $16 on a boxed trifecta of ranks 6-10 (Shockletz, Juja Kibo, Enfjaar, Hopewell Rock, Stinger Glass; 60 combinations, roughly a 27% flexi).

Backtest: returned $179 / $0 / $147 / $29 / $0 / $30 over the six Cups. Paid something in 4 of 6, but only 2 of 6 returned more than $100 at exchange prices, and ROI was -36% (exchange) / -44% (bookmaker). The trifecta slice itself never hit in six Cups (and its dividends are simulated).

## After the barrier draw (Sat 31 Oct) and on race morning

1. Re-rank the field by the price you can actually get on your site; re-pick ranks 8-17 (or 4-6 and 6-10 for Card B).
2. Drop scratched horses and put their stake on the next rank down (a scratching after the draw isn't replaced).
3. Place bets before 3:00pm AEDT. No top-ups after the race: the $100 is the whole budget.
4. A frozen model prediction (`freeze-2026-cup` tag) will be added after the draw; it uses market-only probabilities, so it will agree with the price ranking above.

## Card C: the $1,000 shot (all or nothing)

To return at least $1,000 on a win bet you need stake x price >= $1,000. The chance of a horse winning is about 1/price less the bookmaker margin, so every dollar staked buys the same sliver of that chance (about 1/1,150 per dollar at a 15% margin). With $100 the ceiling is roughly an 8-9% chance of a $1,000+ return and about a 91% chance of losing the $100. How the $100 is split only changes how many horses you cheer for. Longshots are usually over-bet, so the true chance may be lower.

Backtest context (2020-2025): one winner in six was at $51 or longer (Knight's Choice 2024, Betfair $268), so a $1,000 return would have happened once in six Cups for a spread like this. In-sample, six races, no edge.

| Horse | Price now | Stake | Pays if it wins |
|---|---|---|---|
| Strictly Business | $201 | $5 | $1,005 |
| Asterix (NZ) | $101 | $10 | $1,010 |
| Onesmoothoperator (USA) | $101 | $10 | $1,010 |
| Royal Supremacy (IRE) | $101 | $10 | $1,010 |
| Knight's Choice | $51 | $20 | $1,020 |
| Soul Of Spain (IRE) | $51 | $20 | $1,020 |
| Wine Dark Sea (GB) | $51 | $20 | $1,020 |

Total staked $95; keep $5 as a buffer. Rule after the draw: for each horse stake = $1,000 divided by its current price, rounded up; if a price shortens, top up from the buffer; if a horse is scratched or does not make the field, move its stake to the next longest price. Horses at $101 and longer may not make the final 24, so check the field first. Expected loss is similar to Card A in dollars (about $15-$45 of the $100); the difference is that this card loses almost every time and pays big when it does.
