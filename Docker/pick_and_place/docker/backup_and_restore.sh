
mongosh "mongodb://root:example@localhost:27017/?authSource=admin"
db.runCommand( { flushRouterConfig: 1 } )

mongodump "mongodb://root:example@localhost:27017/?authSource=admin" --archive=/tmp/jobs.dump --db=local --collection=jobs
mongodump "mongodb://root:example@localhost:27017/?authSource=admin" --archive=/tmp/models.dump --db=local --collection=models
mongodump "mongodb://root:example@localhost:27017/?authSource=admin" --archive=/tmp/lb.dump --db=local --collection=leaderboard_scores

mongodump "mongodb://root:example@localhost:27017/?authSource=admin" --archive=/data/db/backups/jobs.dump --db=local --collection=jobs
mongodump "mongodb://root:example@localhost:27017/?authSource=admin" --archive=/data/db/backups/models.dump --db=local --collection=models
mongodump "mongodb://root:example@localhost:27017/?authSource=admin" --archive=/data/db/backups/lb.dump --db=local --collection=leaderboard_scores

mongorestore "mongodb://root:example@localhost:27017/?authSource=admin" --verbose --nsFrom=local.jobs --nsTo=robotaxi.jobs --archive=/tmp/jobs.dump
mongorestore "mongodb://root:example@localhost:27017/?authSource=admin" --verbose --nsFrom=local.models --nsTo=robotaxi.models --archive=/tmp/models.dump
mongorestore "mongodb://root:example@localhost:27017/?authSource=admin" --verbose --nsFrom=local.leaderboard_scores --nsTo=robotaxi.leaderboard_scores --archive=/tmp/lb.dump

mongorestore "mongodb://root:example@mongo:27017/?authSource=admin" --verbose --nsFrom=local.jobs --nsTo=robotaxi.jobs --archive=/data/db/backups/jobs.dump
mongorestore "mongodb://root:example@mongo:27017/?authSource=admin" --verbose --nsFrom=local.models --nsTo=robotaxi.models --archive=/data/db/backups/models.dump
mongorestore "mongodb://root:example@mongo:27017/?authSource=admin" --verbose --nsFrom=local.leaderboard_scores --nsTo=robotaxi.leaderboard_scores --archive=/data/db/backups/lb.dump


db.createUser({ user:'root',pwd:'example',roles:['readWrite', 'dbAdmin', 'userAdmin']})