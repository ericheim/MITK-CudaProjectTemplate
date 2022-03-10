/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateOr.h>
#include <mitkNodePredicateProperty.h>
#include <mitkImage.h>

#include <usModuleRegistry.h>

#include <QMessageBox>

#include <CudaMinMaxScaler.h>
#include <CudaMinMaxScalerStub.h>

#include "QmitkExampleView.h"

// Don't forget to initialize the VIEW_ID.
const std::string QmitkExampleView::VIEW_ID = "org.mitk.views.exampleview";

void QmitkExampleView::CreateQtPartControl(QWidget* parent)
{
  // Setting up the UI is a true pleasure when using .ui files, isn't it?
  m_Controls.setupUi(parent);

  m_Controls.selectionWidget->SetDataStorage(this->GetDataStorage());
  m_Controls.selectionWidget->SetSelectionIsOptional(true);
  m_Controls.selectionWidget->SetEmptyInfo(QStringLiteral("Select an image"));
  m_Controls.selectionWidget->SetNodePredicate(mitk::NodePredicateAnd::New(
    mitk::TNodePredicateDataType<mitk::Image>::New(),
    mitk::NodePredicateNot::New(mitk::NodePredicateOr::New(
      mitk::NodePredicateProperty::New("helper object"),
      mitk::NodePredicateProperty::New("hidden object")))));

  // Wire up the UI widgets with our functionality.
  connect(m_Controls.selectionWidget, &QmitkSingleNodeSelectionWidget::CurrentSelectionChanged, this, &QmitkExampleView::OnImageChanged);
  connect(m_Controls.processImageButton, SIGNAL(clicked()), this, SLOT(ProcessSelectedImage()));

  // Make sure to have a consistent UI state at the very beginning.
  this->OnImageChanged(m_Controls.selectionWidget->GetSelectedNodes());
}

void QmitkExampleView::SetFocus()
{
  m_Controls.processImageButton->setFocus();
}

void QmitkExampleView::OnImageChanged(const QmitkSingleNodeSelectionWidget::NodeList&)
{
  this->EnableWidgets(m_Controls.selectionWidget->GetSelectedNode().IsNotNull());
}

void QmitkExampleView::EnableWidgets(bool enable)
{
  m_Controls.processImageButton->setEnabled(enable);
}

void QmitkExampleView::ProcessSelectedImage()
{
  using Scaler = mitk::cuda_example::CudaMinMaxScaler;

  auto selectedDataNode = m_Controls.selectionWidget->GetSelectedNode();
  auto data = selectedDataNode->GetData();

  // We don't use the auto keyword here, which would evaluate to a native
  // image pointer. Instead, we want a smart pointer to ensure that
  // the image isn't deleted somewhere else while we're using it.
  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(data);

  auto imageName = selectedDataNode->GetName();

  MITK_INFO << "Scale image \"" << imageName << "\" ...";

  mitk::cuda_example::CudaMinMaxScalerStub stub;
  Scaler::Pointer scaler = Scaler::New(stub);
  scaler->SetInput(image);
  scaler->Update();

  mitk::Image::Pointer scaledImage = scaler->GetOutput();

  if (scaledImage.IsNull() || !scaledImage->IsInitialized())
    return;

  MITK_INFO << "  done";

  // Stuff the resulting image into a data node, set some properties,
  // and add it to the data storage, which will eventually display the
  // image in the application.
  auto scaledImageDataNode = mitk::DataNode::New();
  scaledImageDataNode->SetData(scaledImage);

  const QString name = QString("%1_scaled").arg(imageName.c_str());
  scaledImageDataNode->SetName(name.toStdString());

  this->GetDataStorage()->Add(scaledImageDataNode);
}
