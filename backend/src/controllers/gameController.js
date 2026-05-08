const GameScore = require('../models/GameScore');
const User = require('../models/User');

exports.saveScore = async (req, res) => {
  const { score } = req.body;
  try {
    const newScore = new GameScore({
      userId: req.user.id,
      score
    });
    await newScore.save();

    // Add XP to user based on score
    // Basic logic: 1 XP per 10 points score
    const xpEarned = Math.floor(score / 10);
    const user = await User.findById(req.user.id);
    user.xp += xpEarned;
    
    // Auto level up logic (every 100 XP = 1 level)
    user.level = Math.floor(user.xp / 100) + 1;
    await user.save();

    res.json({ message: 'Score saved successfully', xpEarned, newTotalXp: user.xp, level: user.level });
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server error');
  }
};

exports.getScores = async (req, res) => {
  try {
    const scores = await GameScore.find({ userId: req.user.id }).sort({ createdAt: -1 }).limit(10);
    res.json(scores);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server error');
  }
};
