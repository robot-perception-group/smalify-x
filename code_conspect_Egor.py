# smplify-x

# main.py

fit_single_frame()


# fit_single_frame.py

def fit_single_frame()
  # set up
  init_t = fitting.guess_init()
  camera_loss = fitting.create_loss()
  loss = fitting.create_loss()
  with fitting.FittingMonitor() as monitor:
    with torch.no_grad():
      body_model.betas[:] = torch.Tensor([zebra_betas])
    camera_optimizer, camera_create_graph = optim_factory.create_optimizer()
    fit_camera = monitor.create_fitting_closure()
    # Step 1
    cam_init_loss_val = monitor.run_fitting(camera_optimizer,fit_camera)
    with torch.no_grad():
      camera.translation[:] = forced_translation.view_as(camera.translation)
      body_model.global_orient[:] = forced_orient.view_as(body_model.global_orient)
    # Step 2
    for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
      for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
        body_optimizer, body_create_graph = optim_factory.create_optimizer()
        closure = monitor.create_fitting_closure( body_optimizer, body_model)
        final_loss_val = monitor.run_fitting(body_optimizer,closure)


# fitting.py

def guess_init()
class FittingMonitor(object)
  def run_fitting()
    for n in range(self.maxiters):
      loss = optimizer.step(closure)
  def create_fitting_closure()
class SMPLifyLoss(nn.Module)
  def forward()
class SMPLifyCameraInitLoss(nn.Module):
  def forward()




# why use SMPLOutput class instead of storing everything in SMPL?
# consider vertex_joint_selector