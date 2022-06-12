echo "Committing to Git.."
git add bottleneck.py  data.py  decoder.py  git_push.sh  metrics.py  pred_utils.py  run.sh callbacks.sh   data.sh  encoder.py  main.py      modules.sh  results.py     unet_model.py  viz_utils.py Pet-Image-Segmentor/model
git commit -m "Main Push"
git remote add origin "https://github.com/Goldenprince8420/Pet-Image-Segmentor.git"
git remote -v
git push origin main
echo "Commit Done!!"
