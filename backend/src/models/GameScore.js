const mongoose = require('mongoose');

const GameScoreSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  score: { type: Number, required: true }
}, { timestamps: true });

module.exports = mongoose.model('GameScore', GameScoreSchema);
