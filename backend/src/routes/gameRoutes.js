const express = require('express');
const router = express.Router();
const gameController = require('../controllers/gameController');
const auth = require('../middleware/auth');

router.post('/score', auth, gameController.saveScore);
router.get('/scores', auth, gameController.getScores);

module.exports = router;
