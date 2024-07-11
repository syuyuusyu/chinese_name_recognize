v="prod"
ip="swr.cn-north-1.myhuaweicloud.com"
name="bqm/name_recognize_flask_app"
docker build -t $ip/$name:$v .
docker push $ip/$name:$v &&
echo $ip/$name:$v