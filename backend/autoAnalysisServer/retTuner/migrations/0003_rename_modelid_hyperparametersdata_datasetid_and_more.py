# Generated by Django 4.2.7 on 2024-07-04 11:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('retTuner', '0002_datasetsdata_hyperparametersdata_modeldata_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='hyperparametersdata',
            old_name='modelID',
            new_name='datasetID',
        ),
        migrations.RemoveField(
            model_name='hyperparametersdata',
            name='description',
        ),
        migrations.AddField(
            model_name='hyperparametersdata',
            name='value',
            field=models.CharField(default='lol', max_length=500),
            preserve_default=False,
        ),
    ]
