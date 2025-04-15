# Chess AI, still quite stupid

## How it works 
I trained a bot on a dataset of matches from TWIC, storing only the matches with 2k elo or above. I trained the bot with two policy heads, and a value head. The policy returns two distributions of probabilities over 64 squares, one for the start square 
and one for the end square. The value head is meant to predict the outcome of the game, trained through MSE loss and outputs a value from -1 to 1 (tanh). 

### Model architecture
THe model takes an FEN postiion, and converts it into a (17, 8, 8) tensor with information about each piece, castling rights, and many more things. It is then fed into a CNN and flattened, and has linear connections to all heads.


## Results after training
Over the course of a few days, I trained the bot to around 50-60 epochs on the TWIC dataset (800k-ish positions). 

Then, I attempted a selfplay cycle, using MCTS and an AlphaZero inspired training cycle. Unfortunately, although the bot ran quite quickly at 
50 simulations per turn, upping that number to a more trainable 1000-1500 left each match creation taking around 15 minutes on my personal computer (google colab would work sometimes, however production times were still pretty high and I didn't always get to progress >9 epochs in order to save
any meaningful progress). Thus, I still trained it on 100-200ish simulations per turn for around 300 epochs, however the actual effects of this training was minute (value head still needed some working on in order to be of any effect in the MCTS)

Sometime along the way, I attempted adding MCTS batching, a weird idea that turned out horribly as now the bot was playing much worse than before. Thanks to this, though, I was able to capture a very odd match played by the bot, pgn shown below:


Nh3 Nh6 2. Ng5 Rg8 3. Nxh7 Rh8 4. Nxf8 Rg8 5. Nh7 Rh8 6. Nf8 Rg8 7. Nh7 Rh8 8. Nf8 Rg8 9. Nh7 Rh8 10. Nf8 Rg8 11. Nh7 Rh8 12. f3 Rg8 13. Nf8 Rh8 14. Nh7 Rg8 15. Nf8 Rh8 16. Nh7 Rg8 17. Nf8 Rh8 18. Nh7 Rg8 19. Nf8 Rh8 20. e4 Rg8 21. Nh7 Rh8 22. Nf8 Rg8 23. Nh7 Rh8 24. Nf8 Rg8 25. Nh7 Rh8 26. Nf8 Rg8 27. Nh7 Rh8 28. d4 Rg8 29. Nf8 Rh8 30. Nh7 Rg8 31. Nf8 Rh8 32. Nh7 Rg8 33. Nf8 Rh8 34. Nh7 Rg8 35. Nf8 Rh8 36. Bc4 Rg8 37. Nh7 Rh8 38. Nf8 Rg8 39. Nh7 Rh8 40. Nf8 Rg8 41. Nh7 Rh8 42. Nf8 Rg8 43. Nh7 Rh8 44. O-O Rg8 45. Nf8 Rh8 46. Nh7 Rg8 47. Nf8 Rh8 48. Nh7 Rg8 49. Nf8 Rh8 50. Nh7 Rg8 51. Nf8 Rh8 52. a3 Rg8 53. Nh7 Rh8 54. Nf8 Rg8 55. Nh7 Rh8 56. Nf8 Rg8 57. Nh7 Rh8 58. Nf8 Rg8 59. Nh7 Rh8 60. c3 Rg8 61. Nf8 Rh8 62. Nh7 Rg8 63. Nf8 Rh8 64. Nh7 Rg8 65. Nf8 Rh8 66. Nh7 Rg8 67. Nf8 Rh8 68. Be2 Rg8 69. Nh7 Rh8 70. Nf8 Rg8 71. Nh7 Rh8 72. Nf8 Rg8 73. Nh7 Rh8 74. Nf8 Rg8 75. Nh7 Rh8 76. Qe1 Rg8 77. Nf8 Rh8 78. Nh7 Rg8 79. Nf8 Rh8 80. Nh7 Rg8 81. Nf8 Rh8 82. Nh7 Rg8 83. Nf8 Rh8 84. g3 Rg8 85. Nh7 Rh8 86. Nf8 Rg8 87. Nh7 Rh8 88. Nf8 Rg8 89. Nh7 Rh8 90. Nf8 Rg8 91. Nh7 Rh8 92. Qd2 Rg8 93. Nf8 Rh8 94. Nh7 Rg8 95. Nf8 Rh8 96. Nh7 Rg8 97. Nf8 Rh8 98. Nh7 Rg8 99. Nf8 Rh8 100. Kh1 Rg8 101. Nh7 Rh8 102. Nf8 Rg8 103. Nh7 Rh8 104. Nf8 Rg8 105. Nh7 Rh8 106. Nf8 Rg8 107. Nh7 Rh8 108. c4 Rg8 109. Nf8 Rh8 110. Nh7 Rg8 111. Nf8 Rh8 112. Nh7 Rg8 113. Nf8 Rh8 114. Nh7 Rg8 115. Nf8 Rh8 116. Qc3 Rg8 117. Nh7 Rh8 118. Nf8 Rg8 119. Nh7 Rh8 120. Nf8 Rg8 121. Nh7 Rh8 122. Nf8 Rg8 123. Nh7 Rh8 124. Nf6+ Kf8 125. Ng8 Rxg8 126. Qa5 Rh8 127. Qxc7 Rg8 128. Qxd8# *

And in another match it tied, unable to see proper moves during the endgame. I don't have the exact PGN, however running the model with around 1500-2000 simulations for selfplay will almost always resul tin a draw, as it starts strongly but quickly deteriorates quality. 

Due to this, I decided to add 100k more positions to my main training file, in order to further diversify the data and hopefully allow the bot to understand positions more. Additionally, I created a new file just for endgames, filling it with around 300k positions for the bot to train on. I left the bot to train (mainly on the endgame positions)
for around 90-100 epochs, and it could now finish a small number of games against itself (it was at this point that I realized dirilecht noise was required to make my MCTS not output the same value every time I generated a selfplay game, since I wasn't using a random selector).


By the end, the bot still sucks at playing endgames, an issue I suspect can mainly be fixed through selfplay, however I have neither the resources nor the time to continue trianing the model (it can be good, it just needs time, which I don't have). It has learnt to identify forks, however, and can see basic 1-2 (maybe 3?) move combos/traps

## Favorite openings:
As white, it loves the Ruy Lopez (assuming e4, e5, Nf3, Nc6) or the London. As black, it prefers the Caro-Kann, or the Sicilian (both are pretty similar openings in terms of starting square). An interesting thing is that it plays the top move according to stockfish every time for the first 5-6 moves but then it goes on it's own tangent, which makes sense since the grandmasters it trains on most likely train themselves on the best moves under their specific opening.
## Further training
A clever idea I came across was to train the bot on chess puzzles with multi-move solutions and trian on each move, this could allow it to tohink a lot more intuitively about tricks and nuances within its gameplay

